"""Actual meta-model counts and explicit lower-bound resource admission.

Counting constructs the pinned Hugging Face model on PyTorch's meta device. It
never loads pretrained tensors or allocates their storage. Memory numbers describe
the current FP32 stage/CPU-Adam runtime; they are not training peak measurements.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Mapping


def measure_model_capacity(
    config: Mapping[str, Any], num_stages: int, mode: str = "lora", sequence_length: int = 64,
) -> dict[str, Any]:
    """Count exact model/stage parameters with no materialized model weights.

    ``config`` may be the text config, a multimodal config with text_config, or a
    run manifest's model object containing config. Counts include every expert;
    structural active parameters are *not* used for weight-storage sizing.

    Memory assumptions: batch=1, FP32 model/gradients, CPU AdamW m/v (8 bytes per
    trainable parameter), and accelerator workers' additional FP32 CPU master and
    CPU gradient copy (8 more bytes). Conservative checkpoints include the full
    model because adapter-only snapshots require separately verified pretrained
    base weights. Smaller adapter-only alternatives are reported explicitly.
    """
    if mode not in {"lora", "full"}:
        raise ValueError("mode must be lora or full")
    if type(sequence_length) is not int or sequence_length < 2:
        raise ValueError("sequence_length must be an integer >= 2")
    if not isinstance(config, Mapping):
        raise ValueError("config must be an object")
    raw = dict(config.get("config", config))
    from .model import QwenStage, _hf, _text_dict, inject_lora
    import torch

    text = _text_dict(raw)
    layers = text.get("num_hidden_layers")
    if type(layers) is not int or layers <= 0 or type(num_stages) is not int or not 1 <= num_stages <= layers:
        raise ValueError("num_stages must be between one and the decoder layer count")
    if text.get("tie_word_embeddings", False) and num_stages > 1:
        raise ValueError("Current distributed runtime requires untied cross-stage embeddings")
    Model, Config, _ = _hf()
    hf_config = Config(**text)
    hf_config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Model(hf_config)
        base_parameters = sum(parameter.numel() for parameter in model.parameters())
        if mode == "lora":
            options = raw.get("lora", {})
            inject_lora(model, rank=options.get("rank", options.get("r", 16)),
                        alpha=options.get("alpha", 32), dropout=options.get("dropout", 0), seed=0)
        else:
            model.requires_grad_(True)
        stages = [QwenStage(model, index, num_stages) for index in range(num_stages)]
    if any(parameter.device.type != "meta" for parameter in model.parameters()):
        raise RuntimeError("Capacity counting unexpectedly materialized a model parameter")

    experts, top_k = hf_config.num_experts, hf_config.num_experts_per_tok
    if not 1 <= top_k <= experts:
        raise ValueError("Require 1 <= num_experts_per_tok <= num_experts")
    stage_reports = []
    for index, stage in enumerate(stages):
        named = list(stage.named_parameters())
        parameters = sum(parameter.numel() for _, parameter in named)
        trainable = sum(parameter.numel() for _, parameter in named if parameter.requires_grad)
        routed = sum(parameter.numel() for name, parameter in named if ".mlp.experts." in "." + name)
        shared = parameters - routed
        structural_active = shared + routed * top_k / experts
        model_bytes, gradients, adam, master = parameters * 4, trainable * 4, trainable * 8, trainable * 4
        # state_dict intentionally counts duplicate keys (e.g. tied single-stage
        # embedding/head): cpu_tree clones each serialized entry separately.
        model_state_bytes = sum(tensor.numel() * (4 if tensor.is_floating_point() else tensor.element_size())
                                for tensor in stage.state_dict().values())
        adapter_state_bytes = sum(parameter.numel() * 4 for name, parameter in named
                                  if name.endswith(("lora_A", "lora_B")))
        cpu_checkpoint = model_state_bytes + adam
        accelerator_checkpoint = model_state_bytes + adam + master
        hidden_bytes = sequence_length * hf_config.hidden_size * 4
        local_layers = stage.layer_range[1] - stage.layer_range[0]
        routers = sequence_length * experts * local_layers * 4
        logits = sequence_length * hf_config.vocab_size * 4 if stage.is_tail else 0
        activation = 2 * hidden_bytes + routers + logits
        # This is deliberately only a lower bound: attention matrices, expert
        # dispatch, reference DeltaNet intermediates, allocator fragmentation and
        # RPC serialization buffers are not derivable from parameter counts.
        cpu_runtime = model_bytes + gradients + adam + activation
        accelerator_runtime = model_bytes + gradients + activation
        cpu_offload_runtime = master + adam + gradients + 2 * hidden_bytes + routers
        largest_tensor = max(parameter.numel() * 4 for _, parameter in named)
        minimum_cpu_host = max(cpu_runtime, model_bytes + gradients + adam + cpu_checkpoint)
        minimum_accelerator_host = max(cpu_offload_runtime, master + adam + gradients + accelerator_checkpoint,
                                       master + adam + largest_tensor)
        stage_reports.append({
            "stage_index": index, "layer_range": list(stage.layer_range),
            "parameters": parameters, "trainable_parameters": trainable, "routed_parameters": routed,
            "shared_parameters": shared, "structural_active_parameters": structural_active,
            "cpu_model_bytes_fp32": model_bytes, "optimizer_state_bytes": adam,
            "optimizer_master_bytes": master, "gradients_bytes": gradients,
            "additional_optimizer_master_bytes_cpu_worker": 0,
            "additional_optimizer_master_bytes_accelerator_worker": master,
            "cpu_gradient_copy_bytes_accelerator_worker": gradients,
            "activation_lower_bound_bytes": activation,
            "activation_components": {"two_hidden_boundaries_bytes": 2 * hidden_bytes,
                                      "router_logits_bytes": routers, "tail_logits_bytes": logits},
            "cpu_training_lower_bound_bytes": cpu_runtime,
            "accelerator_training_lower_bound_bytes": accelerator_runtime,
            "accelerator_cpu_offload_lower_bound_bytes": cpu_offload_runtime,
            "minimum_host_bytes": minimum_cpu_host,
            "minimum_accelerator_bytes": accelerator_runtime,
            "accelerator_worker_minimum_host_bytes": minimum_accelerator_host,
            "worker_checkpoint_copy_bytes_cpu": cpu_checkpoint,
            "worker_checkpoint_copy_bytes_accelerator": accelerator_checkpoint,
            "checkpoint_full_model_bytes": model_state_bytes,
            "checkpoint_adapter_model_bytes": adapter_state_bytes,
            "adapter_only_checkpoint_bytes_cpu": adapter_state_bytes + adam,
            "adapter_only_checkpoint_bytes_accelerator": adapter_state_bytes + adam + master,
            "largest_parameter_bytes_fp32": largest_tensor,
            "parameter_storage_device": "meta", "measured_training_peak_bytes": None,
        })
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if sum(stage["parameters"] for stage in stage_reports) != parameters:
        raise RuntimeError("Stage parameter ownership does not match the full model")
    routed = sum(stage["routed_parameters"] for stage in stage_reports)
    checkpoint_cpu = sum(stage["worker_checkpoint_copy_bytes_cpu"] for stage in stage_reports)
    checkpoint_accelerator = sum(stage["worker_checkpoint_copy_bytes_accelerator"] for stage in stage_reports)
    adapter_only = sum(stage["adapter_only_checkpoint_bytes_accelerator"] for stage in stage_reports)
    experiment = raw.get("_sakura_experiment", {})
    return {
        "schema_version": 1, "count_method": "actual HF Qwen3.5 meta-model and owned QwenStage partitions",
        "parameter_counts_exact_for_config": True, "materialized_model_weight_bytes": 0,
        "config_sha256": hashlib.sha256(json.dumps({"text_config": text, "mode": mode,
                            "lora": raw.get("lora", {}) if mode == "lora" else None},
                            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "mode": mode, "num_stages": num_stages, "batch_size": 1, "sequence_length": sequence_length,
        "base_parameters": base_parameters, "total_parameters": parameters,
        "trainable_parameters": sum(stage["trainable_parameters"] for stage in stage_reports),
        "routed_parameters": routed, "shared_parameters": parameters - routed,
        "structural_active_parameters": parameters - routed + routed * top_k / experts,
        "num_experts": experts, "top_k": top_k, "stages": stage_reports,
        "cpu_model_bytes_fp32": parameters * 4,
        "cpu_training_lower_bound_bytes": sum(stage["cpu_training_lower_bound_bytes"] for stage in stage_reports),
        "checkpoint_bytes_cpu_workers": checkpoint_cpu,
        "checkpoint_bytes_accelerator_workers": checkpoint_accelerator,
        "checkpoint_adapter_only_bytes_accelerator_workers": adapter_only if mode == "lora" else None,
        "coordinator_checkpoint_gather_lower_bound_bytes": checkpoint_accelerator,
        "coordinator_checkpoint_memory_profile": "conservative accelerator CPU-master snapshots; CPU-only snapshot alternative is checkpoint_bytes_cpu_workers",
        "checkpoint_disk_bytes_per_copy": 2 * checkpoint_accelerator,
        "checkpoint_policy": "conservative full-state snapshots; adapter-only figures require verified pinned base weights",
        "checkpoint_disk_generations": 2,
        "checkpoint_retention_note": "two generations is an atomic-update minimum; current runtime retains additional generations until explicitly pruned",
        "proxy": experiment.get("proxy"),
        "synthetic_vocabulary": experiment.get("synthetic_vocabulary"),
        "training_peak_measured": False,
        "limits": ["activation and memory figures are lower bounds, not successful-fit predictions",
                   "batch size one; runtime model/gradients and CPU Adam state are FP32",
                   "full routed weights stay resident even when top-k activates a small fraction",
                   "coordinator gathers every stage checkpoint in CPU RAM before writing",
                   "serialization, framework allocations, temporary workspaces and filesystem overhead are additional",
                   "host identities and physical checkpoint independence need measured operator evidence"],
    }


def _bytes(value: Any, description: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{description} must be nonnegative measured integer bytes")
    return value


def admit_run(
    report: Mapping[str, Any], doctor_reports: Iterable[Mapping[str, Any]], checkpoint_copies: int = 2,
    *, coordinator_report: Mapping[str, Any] | None = None, margin: float = 0.30,
) -> dict[str, Any]:
    """Check measured budgets against lower bounds and propose stage placement.

    Doctor reports follow doctor.probe's schema. Optional host_id values establish
    which measurements refer to the same physical host; they must be assigned and
    verified by operators. A coordinator may share a worker host by using its same
    host_id. Explicit coordinator_report (or one row marked role=coordinator) is
    mandatory because the current trainer gathers *all* snapshots in CPU RAM.

    ``feasible`` reports only that these lower-bound checks passed with 30% of
    measured free capacity reserved. It never certifies actual peak training fit.
    MPS unified memory consumes the host budget for both device and CPU offload.
    """
    if type(checkpoint_copies) is not int or checkpoint_copies < 2:
        raise ValueError("at least two checkpoint copies on different physical hosts are required")
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or not 0 <= margin < 1:
        raise ValueError("margin must be finite and in [0, 1)")
    stages = list(report.get("stages", []))
    if not stages:
        raise ValueError("a measured model capacity report with stages is required")
    doctors = [dict(doctor) for doctor in doctor_reports]
    if not doctors:
        return {"feasible": False, "status": "requires_worker_measurements", "placements": [], "margin": margin,
                "training_fit_certified": False}
    marked = [doctor for doctor in doctors if doctor.get("role") == "coordinator"]
    if coordinator_report is None and len(marked) == 1:
        coordinator_report = marked[0]
    # Host entries are merged by explicit physical identity, never by directory,
    # platform name, member ID, GPU name or coincident memory numbers.
    hosts: list[dict[str, Any]] = []
    host_by_id: dict[str, int] = {}

    def add_host(doctor: Mapping[str, Any], role: str) -> int:
        identity = doctor.get("host_id")
        if identity is not None and (not isinstance(identity, str) or not identity.strip()):
            raise ValueError("host_id must be a nonempty physical host identifier")
        available = _bytes(doctor.get("memory", {}).get("available_bytes"), "memory.available_bytes")
        disk = _bytes(doctor.get("disk", {}).get("free_bytes"), "disk.free_bytes")
        if identity is not None and identity in host_by_id:
            index = host_by_id[identity]
            entry = hosts[index]
            entry["host_budget_bytes"] = min(entry["host_budget_bytes"], int(available * (1 - margin)))
            entry["disk_budget_bytes"] = min(entry["disk_budget_bytes"], int(disk * (1 - margin)))
            entry["roles"].add(role)
            return index
        index = len(hosts)
        entry = {"host_id": identity, "report_index": index, "roles": {role},
                 "host_budget_bytes": int(available * (1 - margin)), "disk_budget_bytes": int(disk * (1 - margin)),
                 "host_reserved_bytes": 0, "disk_reserved_bytes": 0}
        hosts.append(entry)
        if identity is not None:
            host_by_id[identity] = index
        return index

    workers = []
    device_pools: dict[tuple[int, str], dict[str, int | None]] = {}
    for index, doctor in enumerate(doctors):
        if doctor.get("role") == "coordinator":
            continue
        host = add_host(doctor, "worker")
        backend = doctor.get("backend", str(doctor.get("device", "cpu")).split(":", 1)[0])
        if backend not in {"cpu", "cuda", "rocm", "mps"}:
            raise ValueError("doctor backend must be cpu, cuda, rocm or mps")
        free = doctor.get("accelerator", {}).get("free_bytes")
        device = doctor.get("device", backend)
        if not isinstance(device, str) or not device:
            raise ValueError("doctor device must be a nonempty string")
        budget = int(_bytes(free, "accelerator.free_bytes") * (1 - margin)) if free is not None else None
        # probe runs in a fresh CLI process, whose implicit CUDA device is zero.
        # Normalize the alias so cuda and cuda:0 do not invent two GPUs.
        device_key = (host, "cuda:0" if device in {"cuda", "rocm"} else device)
        if device_key not in device_pools:
            device_pools[device_key] = {"budget": budget, "reserved": 0}
        elif budget is not None:
            previous = device_pools[device_key]["budget"]
            device_pools[device_key]["budget"] = min(previous, budget) if previous is not None else budget
        workers.append({"doctor_index": index, "host_index": host, "backend": backend,
                        "device_pool": device_key, "device": device})
    coordinator_host = add_host(dict(coordinator_report), "coordinator") if coordinator_report is not None else None
    problems, unknowns = [], []
    if coordinator_host is None:
        unknowns.append("coordinator host RAM/disk measurement is required; current trainer gathers all stage checkpoints")
    if not workers:
        problems.append("no worker doctor reports were supplied")
    # Reserve the worst-case accelerator checkpoint shape before placing workers
    # so a shared coordinator host cannot spend the same RAM twice.
    coordinator_bytes = _bytes(report.get("coordinator_checkpoint_gather_lower_bound_bytes"), "coordinator gather bytes")
    if coordinator_host is not None:
        hosts[coordinator_host]["host_reserved_bytes"] += coordinator_bytes
        if hosts[coordinator_host]["host_reserved_bytes"] > hosts[coordinator_host]["host_budget_bytes"]:
            problems.append("coordinator cannot hold the aggregate checkpoint within its measured RAM budget")
    placements = []
    for stage in sorted(stages, key=lambda row: -row["minimum_host_bytes"]):
        candidates = []
        for worker_index, worker in enumerate(workers):
            host = hosts[worker["host_index"]]
            pool = device_pools[worker["device_pool"]]
            if worker["backend"] == "cpu":
                host_bytes, accelerator_bytes = stage["minimum_host_bytes"], 0
            else:
                host_bytes = stage["accelerator_worker_minimum_host_bytes"]
                accelerator_bytes = stage["minimum_accelerator_bytes"]
                if pool["budget"] is None:
                    continue
                if pool["reserved"] + accelerator_bytes > pool["budget"]:
                    continue
                if worker["backend"] == "mps":
                    host_bytes += accelerator_bytes
            if host["host_reserved_bytes"] + host_bytes <= host["host_budget_bytes"]:
                remaining = host["host_budget_bytes"] - host["host_reserved_bytes"] - host_bytes
                candidates.append((remaining, worker_index, host_bytes, accelerator_bytes))
        if not candidates:
            if any(w["backend"] != "cpu" and device_pools[w["device_pool"]]["budget"] is None for w in workers):
                unknowns.append(f"stage {stage['stage_index']} needs measured accelerator.free_bytes; recommended/total memory is not free memory")
            else:
                problems.append(f"no measured host/device budget can hold stage {stage['stage_index']} lower bound")
            continue
        _, chosen, host_bytes, accelerator_bytes = max(candidates, key=lambda item: (item[0], -item[1]))
        worker = workers[chosen]
        hosts[worker["host_index"]]["host_reserved_bytes"] += host_bytes
        device_pools[worker["device_pool"]]["reserved"] += accelerator_bytes
        placements.append({"stage_index": stage["stage_index"], "doctor_index": worker["doctor_index"],
                           "host_id": hosts[worker["host_index"]]["host_id"], "backend": worker["backend"],
                           "host_required_bytes": host_bytes, "accelerator_required_bytes": accelerator_bytes})
    disk_per_copy = _bytes(report.get("checkpoint_disk_bytes_per_copy"), "checkpoint disk bytes per copy")
    copy_hosts = []
    # The primary checkpoint is always written on the coordinator's filesystem.
    if coordinator_host is not None:
        if hosts[coordinator_host]["disk_budget_bytes"] >= disk_per_copy:
            copy_hosts.append(coordinator_host)
            hosts[coordinator_host]["disk_reserved_bytes"] = disk_per_copy
        else:
            problems.append("coordinator disk cannot hold old and new checkpoint generations with the requested margin")
    for index in sorted(range(len(hosts)), key=lambda i: -hosts[i]["disk_budget_bytes"]):
        if len(copy_hosts) >= checkpoint_copies:
            break
        if index in copy_hosts or hosts[index]["disk_budget_bytes"] < disk_per_copy:
            continue
        identity = hosts[index]["host_id"]
        if identity is None or any(hosts[other]["host_id"] == identity for other in copy_hosts):
            continue
        copy_hosts.append(index)
        hosts[index]["disk_reserved_bytes"] = disk_per_copy
    if coordinator_host is not None and hosts[coordinator_host]["host_id"] is None:
        unknowns.append("coordinator physical host_id is required to verify independent checkpoint copies")
    if len(copy_hosts) < checkpoint_copies:
        if any(host["host_id"] is None for host in hosts):
            unknowns.append("checkpoint copies need measured disk on distinct explicitly identified physical hosts")
        else:
            problems.append(f"fewer than {checkpoint_copies} distinct measured hosts can hold checkpoint copies")
    if coordinator_host is None:
        status = "requires_coordinator_measurement"
    elif problems:
        status = "insufficient_measured_capacity"
    elif unknowns:
        status = "requires_additional_measurements"
    else:
        status = "lower_bound_checks_passed"
    return {"feasible": not problems and not unknowns and len(placements) == len(stages), "status": status,
            "training_fit_certified": False, "training_peak_measured": False, "margin": margin,
            "placements": sorted(placements, key=lambda row: row["stage_index"]),
            "coordinator_checkpoint_gather_required_bytes": coordinator_bytes,
            "checkpoint_copies": checkpoint_copies, "checkpoint_generations_per_copy": 2,
            "checkpoint_disk_required_bytes_per_copy": disk_per_copy,
            "checkpoint_disk_required_bytes_aggregate": checkpoint_copies * disk_per_copy,
            "checkpoint_copy_hosts": [hosts[index]["host_id"] for index in copy_hosts],
            "problems": problems, "unknowns": unknowns,
            "host_budgets": [{**host, "roles": sorted(host["roles"])} for host in hosts],
            "note": "passing lower-bound checks only admits a measured-peak training trial; retained generations and 24h/7d sustained operation need separate evidence"}
