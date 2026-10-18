"""Independent community launcher; does not import Pluralis authentication."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import sys
import threading
import time

from .manifest import load_manifest, run_hash, save_manifest


def _json(value, path=None):
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    else:
        print(content, end="", flush=True)


def _members(paths):
    members = {}
    for path in paths or ():
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value["member_id"] in members:
            raise ValueError("Duplicate member ID")
        members[value["member_id"]] = value["public_key"]
    return members


def _init(args):
    from .model import tiny_config
    from .data import load_records, prepare_dataset, dataset_manifest
    destination = Path(args.directory).resolve()
    if (destination / "run.json").exists():
        raise FileExistsError("Run already exists; use a new directory")
    destination.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.data or Path(__file__).parents[2] / "examples" / "data-fixtures.jsonl").resolve()
    if args.objective == "cpt":
        from .corpus import load_cpt_records
        records = load_cpt_records(dataset_path)
    else:
        records = load_records(dataset_path)
    audit = dataset_manifest(records, prepare_dataset(records))
    if args.profile == "qwen35":
        if not args.base_path or not args.revision or len(args.revision) != 40:
            raise ValueError("qwen35 requires --base-path LOCAL_SNAPSHOT and --revision COMMIT_SHA")
        base = Path(args.base_path).resolve()
        config = json.loads((base / "config.json").read_text(encoding="utf-8"))
        text = config.get("text_config", config)
        text["use_cache"], text["output_router_logits"] = False, True
        model = {"name": "Qwen/Qwen3.5-35B-A3B-Base", "revision": args.revision,
                 "pretrained_path": str(base), "config": config, "seed": args.seed}
        tokenizer = {"kind": "huggingface", "path": str(base), "revision": args.revision}
        from .artifacts import fingerprint_snapshot, fingerprint_tokenizer
        model["artifacts"] = fingerprint_snapshot(base)
        tokenizer["artifacts"] = fingerprint_tokenizer(base)
    elif args.profile == "scratch":
        from .artifacts import fingerprint_tokenizer, verify_tokenizer
        from .tokenization import load_tokenizer
        if args.mode != "full" or not args.config or not args.scratch_tokenizer:
            raise ValueError("scratch requires --mode full --config SELECTED_CONFIG.json --scratch-tokenizer LOCAL_TOKENIZER")
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        text = config.get("text_config", config)
        text["use_cache"], text["output_router_logits"] = False, True
        if args.allow_large_random:
            config["sakura_allow_large_random"] = True
        path = Path(args.scratch_tokenizer).resolve()
        metadata = json.loads((path / "sakura-tokenizer.json").read_text(encoding="utf-8"))
        tokenizer = {"kind": "huggingface", "path": str(path), "revision": metadata["revision"],
                     "artifacts": fingerprint_tokenizer(path)}
        verify_tokenizer(tokenizer, path)
        if len(load_tokenizer(tokenizer)) != text["vocab_size"]:
            raise ValueError("Selected model vocabulary must equal the independently trained tokenizer vocabulary")
        identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        model = {"name": f"SakuraMoE-scratch-{identity[:12]}", "revision": f"random-init-{args.seed}-{identity}",
                 "config": config, "seed": args.seed}
    elif args.profile == "experiment":
        from .experiments import experiment_grid, candidate_to_model_config
        grid = experiment_grid()
        if not 0 <= args.candidate < len(grid):
            raise ValueError("Candidate index must be in [0, 11]")
        config = candidate_to_model_config(grid[args.candidate], proxy=True, synthetic_vocab_size=512)
        model = {"name": f"SakuraMoE-proxy-{grid[args.candidate]['experiment_id']}",
                 "revision": f"random-seed-{args.seed}", "config": config, "seed": args.seed}
        tokenizer = {"kind": "byte", "revision": "fixture-utf8-v1"}
    else:
        config = tiny_config(vocab_size=512, hidden_size=64)
        model = {"name": "qwen35-hybrid-fixture-random", "revision": f"random-seed-{args.seed}",
                 "config": config, "seed": args.seed}
        tokenizer = {"kind": "byte", "revision": "fixture-utf8-v1"}
    manifest = {"schema_version": 1, "run_id": args.run_id or destination.name, "mode": args.mode,
                "model": model, "tokenizer": tokenizer, "num_stages": args.stages,
                "dataset": {"path": str(dataset_path), "revision": audit["data_revision"],
                            "objective": args.objective,
                            "contains_fixtures": audit["contains_fixtures"]},
                "training": {"learning_rate": args.learning_rate, "max_grad_norm": 1.0,
                             "sequence_length": args.sequence_length,
                             "checkpoint_every": args.checkpoint_every or (100 if args.profile == "qwen35" else 1)},
                "members": _members(args.members)}
    save_manifest(destination / "run.json", manifest)
    _json(audit, destination / "dataset-audit.json")
    _json({"manifest": str(destination / "run.json"), "run_hash": run_hash(manifest),
           "profile": args.profile, "fixture_only": audit["contains_fixtures"]})


def _keygen(args):
    from .auth import Identity
    identity = Identity.generate(args.member)
    identity.save(args.private)
    public = Path(args.public)
    if public.exists():
        raise FileExistsError("Public identity destination exists")
    _json({"member_id": identity.member_id, "public_key": identity.public_key}, public)
    _json({"member_id": identity.member_id, "public": str(public.resolve())})


def _certify(device):
    import torch
    from .model import build_model, tiny_config
    from .doctor import resolve_device
    selected = resolve_device(device)
    model = build_model(tiny_config(vocab_size=512, hidden_size=64), seed=0).to(selected)
    model.train()
    frozen = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if not p.requires_grad}
    adapters = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    ids = torch.tensor([[1, 40, 41, 42, 43, 2]], device=selected)
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids, output_router_logits=True,
                       use_cache=False)
        if not torch.isfinite(output.loss):
            raise FloatingPointError("Non-finite training loss")
        output.loss.backward()
        optimizer.step()
        losses.append(float(output.loss.detach()))
    changed = [n for n, p in model.named_parameters() if n in adapters and not torch.equal(adapters[n], p.detach().cpu())]
    unchanged = all(torch.equal(frozen[n], p.detach().cpu()) for n, p in model.named_parameters() if n in frozen)
    if not changed or not unchanged:
        raise RuntimeError("Adapter update/frozen-base certification failed")
    return {"status": "passed", "device": str(selected), "steps": 3, "losses": losses,
            "changed_adapter_tensors": len(changed), "frozen_base_unchanged": unchanged,
            "scope": "small real Qwen3.5 hybrid architecture; not a 35B capacity certification"}


def _doctor(args):
    from .doctor import probe, measure_network
    result = probe(args.directory, args.device, args.available_hours)
    if args.host_id:
        result["host_id"] = args.host_id
    if args.role:
        result["role"] = args.role
    if args.certify:
        result["training_certification"] = _certify(args.device)
    if args.peer_url:
        from .auth import Identity
        from .transport import RpcClient
        manifest = load_manifest(args.manifest)
        client = RpcClient(Identity.load(args.identity), manifest["members"], run_hash(manifest))
        result["network"] = measure_network(client, args.peer_url, args.peer_member)
    _json(result, args.output)


def _inference_trainer(args, manifest):
    from .checkpoint import CheckpointManager
    from .trainer import PipelineTrainer
    trainer = PipelineTrainer(manifest, _endpoints(args, manifest), CheckpointManager(
        args.checkpoints or Path(args.manifest).parent / "checkpoints"), restore_existing=True)
    trainer.resume()
    return trainer


def _infer(args):
    import torch
    from .tokenization import load_tokenizer
    manifest = load_manifest(args.manifest)
    tokenizer = load_tokenizer(manifest["tokenizer"], base_path=args.base_path)
    trainer = _inference_trainer(args, manifest)
    tokens = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}],
                                            tokenize=True, add_generation_prompt=True)
    generated = []
    config = manifest["model"]["config"].get("text_config", manifest["model"]["config"])
    for _ in range(args.max_new_tokens):
        if len(tokens) >= config.get("max_position_embeddings", 2048):
            break
        batch = {"input_ids": torch.tensor([tokens]), "attention_mask": torch.ones((1, len(tokens)), dtype=torch.long)}
        next_token = int(trainer.evaluate_batch(batch)["next_token"][0])
        generated.append(next_token); tokens.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break
    _json({"run_hash": run_hash(manifest), "step": trainer.step_number, "text": tokenizer.decode(generated, skip_special_tokens=True),
           "generated_token_ids": generated, "fixture_only": manifest["dataset"].get("contains_fixtures", False)})


def _score(args):
    from .data import load_records, prepare_dataset, dataset_manifest, encode_windows
    from .tokenization import load_tokenizer
    from .evaluation import bits_per_byte
    manifest = load_manifest(args.manifest)
    from .corpus import load_cpt_records, encode_cpt_windows
    objective = manifest["dataset"].get("objective", "sft")
    records = (load_cpt_records if objective == "cpt" else load_records)(manifest["dataset"]["path"])
    splits = prepare_dataset(records)
    audit = dataset_manifest(records, splits)
    if audit["data_revision"] != manifest["dataset"]["revision"]:
        raise ValueError("Dataset revision changed")
    tokenizer = load_tokenizer(manifest["tokenizer"], base_path=args.base_path)
    trainer = _inference_trainer(args, manifest)
    results = []
    for record in splits[args.split]:
        nll, count = 0.0, 0
        encoder = encode_cpt_windows if objective == "cpt" else encode_windows
        for window in encoder(record, tokenizer, manifest["training"]["sequence_length"]):
            reply = trainer.evaluate_batch(window)
            nll += reply["nll_nats"]; count += reply["supervised_tokens"]
        byte_count = sum(len(message.content.encode("utf-8")) for message in record.messages if message.role == "assistant")
        results.append({"problem_id": record.problem_id, "language": record.language, "nll_nats": nll,
                        "utf8_bytes": byte_count, "supervised_tokens": count,
                        "bits_per_byte": bits_per_byte(nll, byte_count), "executed": False})
    if not results:
        raise ValueError(f"The {args.split} split has no records")
    total_nll, total_bytes = sum(x["nll_nats"] for x in results), sum(x["utf8_bytes"] for x in results)
    _json({"run_hash": run_hash(manifest), "dataset_revision": audit["data_revision"], "split": args.split,
           "step": trainer.step_number, "results": results, "bits_per_byte": bits_per_byte(total_nll, total_bytes),
           "objective": ("raw document next-token NLL including EOS; byte denominator is document UTF-8 content" if objective == "cpt" else
                         "assistant-only next-token NLL including template terminal tokens; byte denominator is assistant UTF-8 content"),
           "fixture_only": audit["contains_fixtures"]}, args.output)


def _batches(manifest, base_path=None, start_cursor=0):
    import torch
    from .data import load_records, prepare_dataset, dataset_manifest, encode_windows
    from .tokenization import load_tokenizer
    from .corpus import load_cpt_records, iter_training_windows
    objective = manifest["dataset"].get("objective", "sft")
    records = (load_cpt_records if objective == "cpt" else load_records)(manifest["dataset"]["path"])
    splits = prepare_dataset(records)
    audit = dataset_manifest(records, splits)
    if audit["data_revision"] != manifest["dataset"]["revision"]:
        raise ValueError("Dataset content/provenance revision differs from the signed run manifest")
    tokenizer = load_tokenizer(manifest["tokenizer"], base_path=base_path)
    config = manifest["model"]["config"].get("text_config", manifest["model"]["config"])
    def batches():
        for window in iter_training_windows(splits["train"], tokenizer, manifest["training"]["sequence_length"],
                                             objective=objective, start_cursor=start_cursor):
            if max(window["input_ids"]) >= config["vocab_size"] or min(window["input_ids"]) < 0:
                raise ValueError("Tokenizer token IDs exceed this model's unchanged vocabulary")
            yield {key: torch.tensor(value, dtype=torch.long).unsqueeze(0) for key, value in window.items()}
    return batches()


def _endpoints(args, manifest):
    from .trainer import StageEndpoint
    if not args.endpoints:
        from .runtime import StageWorker
        from .transport import LocalTransport
        return [StageEndpoint(i, LocalTransport(StageWorker(manifest, i, args.device, base_path=args.base_path).handle).call)
                for i in range(manifest["num_stages"])]
    from .auth import Identity
    from .transport import RpcClient
    client = RpcClient(Identity.load(args.identity), manifest["members"], run_hash(manifest), timeout=args.timeout,
                       max_blob_bytes=args.max_blob_mib * 1024 * 1024)
    configured = json.loads(Path(args.endpoints).read_text(encoding="utf-8"))
    return [StageEndpoint(row["stage_index"],
                          lambda request, row=row: client.call(row["url"], request, expected_member=row["member_id"]))
            for row in configured]


def _train(args):
    from .checkpoint import CheckpointManager
    from .trainer import PipelineTrainer
    manifest = load_manifest(args.manifest)
    from .capacity import measure_model_capacity
    capacity = measure_model_capacity(manifest["model"]["config"], manifest["num_stages"], mode=manifest["mode"],
                                      sequence_length=manifest["training"]["sequence_length"])
    if capacity["base_parameters"] > 500_000_000:
        if not args.admission:
            raise ValueError("Large runs require --admission RESOURCE_REPORT.json from capacity with measured worker/coordinator reports")
        admission = json.loads(Path(args.admission).read_text(encoding="utf-8"))
        if admission.get("run_hash") != run_hash(manifest) or not admission.get("admission", {}).get("feasible", False):
            raise ValueError("Admission report is missing, insufficient, or belongs to another run")
    checkpoint_dir = Path(args.checkpoints or Path(args.manifest).parent / "checkpoints")
    trainer = PipelineTrainer(manifest, _endpoints(args, manifest), CheckpointManager(checkpoint_dir, mirrors=args.mirror),
                              metrics_path=checkpoint_dir.parent / "metrics.jsonl", restore_existing=args.command == "resume")
    if args.command == "resume":
        trainer.resume()
    elif (checkpoint_dir / "CURRENT").exists():
        raise FileExistsError("Checkpoints exist; use resume to avoid replacing training history")
    batches = _batches(manifest, args.base_path, start_cursor=trainer.cursor)
    for _ in range(args.steps):
        metrics = trainer.train_batch(next(batches))
        _json({key: value for key, value in metrics.items() if key != "expert_counts"})
    saved = trainer.checkpoint()
    _json({"checkpoint": str(saved), "step": trainer.step_number, "cursor": trainer.cursor})


def _join(args):
    from .auth import Identity
    from .runtime import StageWorker
    from .transport import RpcServer
    manifest = load_manifest(args.manifest)
    worker = StageWorker(manifest, args.stage, args.device, base_path=args.base_path)
    identity = Identity.load(args.identity)
    server = RpcServer(worker.handle, identity, manifest["members"], run_hash(manifest), host=args.host,
                       port=args.port, allow_external=args.allow_external,
                       max_blob_bytes=args.max_blob_mib * 1024 * 1024).start()
    discovery = None
    stop = threading.Event()
    try:
        if args.initial_peer or args.dht:
            from .discovery import HivemindDiscovery
            discovery = HivemindDiscovery(args.initial_peer, identity, manifest["members"], run_hash(manifest))
            discovery.publish(args.announce_url or server.url)
        _json({"member_id": identity.member_id, "stage_index": args.stage, "url": server.url,
               "run_hash": run_hash(manifest), "transport": "signed-http", "discovery": "hivemind" if discovery else "static"})
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        while not stop.wait(30):
            if discovery:
                discovery.publish(args.announce_url or server.url)
    finally:
        server.close()
        if discovery:
            discovery.close()


def _export(args):
    from .checkpoint import CheckpointManager
    from .model import export_lora_checkpoint
    manifest = load_manifest(args.manifest)
    saved = CheckpointManager(args.checkpoints or Path(args.manifest).parent / "checkpoints").load_current(
        expected_run_hash=run_hash(manifest))
    if manifest["mode"] == "lora":
        export_lora_checkpoint(saved["stages"], manifest["model"]["config"], manifest["num_stages"], args.output,
                               base_model_name_or_path=manifest["model"]["name"], revision=manifest["model"]["revision"])
    else:
        from .exporting import export_full_checkpoint
        export_full_checkpoint(saved["stages"], manifest["model"]["config"], manifest["num_stages"], args.output)
        if manifest["tokenizer"]["kind"] == "huggingface":
            from .artifacts import verify_tokenizer, fingerprint_tokenizer
            source = Path(args.base_path or manifest["tokenizer"]["path"])
            verify_tokenizer(manifest["tokenizer"], source)
            for name in fingerprint_tokenizer(source):
                if name == "config.json":
                    continue  # preserve the new text-only model config
                target = Path(args.output) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, target)
            if (source / "sakura-tokenizer.json").exists():
                shutil.copyfile(source / "sakura-tokenizer.json", Path(args.output) / "sakura-tokenizer.json")
    _json({"run_hash": run_hash(manifest), "step": saved["training_state"]["step"],
           "base_model": {k: v for k, v in manifest["model"].items() if k != "pretrained_path"},
           "tokenizer": {k: v for k, v in manifest["tokenizer"].items() if k != "path"},
           "dataset": {k: v for k, v in manifest["dataset"].items() if k != "path"},
           "fixture_only": manifest["dataset"].get("contains_fixtures", False)}, Path(args.output) / "sakura-run.json")
    _json({"export": str(Path(args.output).resolve()), "step": saved["training_state"]["step"]})


def _prepare_data(args):
    from .data import load_records, prepare_dataset, dataset_manifest
    if args.objective == "cpt":
        from .corpus import load_cpt_records
        records = load_cpt_records(args.input)
    else:
        records = load_records(args.input)
    splits = prepare_dataset(records)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Prepared dataset output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for record in values:
                value = record.to_dict()
                if args.objective == "cpt":
                    value["text"] = value.pop("messages")[0]["content"]
                stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    report = dataset_manifest(records, splits)
    _json(report, output / "manifest.json")
    _json(report)


def _experiments(args):
    from .experiments import experiment_grid, select_experiment
    result = select_experiment(json.loads(Path(args.select).read_text(encoding="utf-8"))) if args.select else experiment_grid()
    _json(result, args.output)


def _tokenizer(args):
    from .experiments import train_tokenizer
    sources = json.loads(Path(args.provenance).read_text(encoding="utf-8"))
    result = train_tokenizer(args.input, args.output, vocab_size=args.vocab_size, provenance=sources,
                             input_format=args.input_format, smoke_test=args.smoke_test)
    _json(result)


def _capacity(args):
    from .capacity import measure_model_capacity, admit_run
    from .experiments import project_schedule
    manifest = load_manifest(args.manifest)
    report = measure_model_capacity(manifest["model"]["config"], manifest["num_stages"], mode=manifest["mode"],
                                    sequence_length=manifest["training"]["sequence_length"])
    report["run_hash"] = run_hash(manifest)
    if args.reports:
        reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.reports]
        coordinator = json.loads(Path(args.coordinator).read_text(encoding="utf-8")) if args.coordinator else None
        report["admission"] = admit_run(report, reports, coordinator_report=coordinator)
    if args.observations:
        if args.target_tokens is None:
            raise ValueError("--target-tokens is required with --observations")
        report["schedule"] = project_schedule(json.loads(Path(args.observations).read_text(encoding="utf-8")),
                                               target_tokens=args.target_tokens)
    _json(report, args.output)


def _evaluate(args):
    from .evaluation import summarize_evaluation, evaluate_python, SandboxLimits
    rows = [json.loads(line) for line in Path(args.results).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.execute:
        limits = SandboxLimits(image=args.image)
        for row in rows:
            execution = evaluate_python(row["code"], row["test_code"], execute=True, limits=limits)
            row.update({"executed": execution.get("executed", False), "passed": execution.get("passed", False),
                        "sandbox": execution})
    baseline = ([json.loads(line) for line in Path(args.baseline).read_text(encoding="utf-8").splitlines() if line.strip()]
                if args.baseline else None)
    _json(summarize_evaluation(rows, baseline=baseline), args.output)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2, help="PyTorch CPU threads (default 2)")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Create an immutable tiny smoke or local pretrained run")
    init.add_argument("--directory", required=True); init.add_argument("--run-id")
    init.add_argument("--profile", choices=["tiny", "qwen35", "experiment", "scratch"], default="tiny")
    init.add_argument("--candidate", type=int, default=0, help="Architecture grid index for --profile experiment")
    init.add_argument("--config"); init.add_argument("--scratch-tokenizer")
    init.add_argument("--allow-large-random", action="store_true")
    init.add_argument("--mode", choices=["lora", "full"], default="lora")
    init.add_argument("--stages", type=int, default=2); init.add_argument("--seed", type=int, default=0)
    init.add_argument("--sequence-length", type=int, default=64)
    init.add_argument("--checkpoint-every", type=int)
    init.add_argument("--learning-rate", type=float, default=0.001)
    init.add_argument("--base-path"); init.add_argument("--revision"); init.add_argument("--data")
    init.add_argument("--objective", choices=["sft", "cpt"], default="sft")
    init.add_argument("--members", nargs="*", default=[]); init.set_defaults(function=_init)
    keys = sub.add_parser("keygen", help="Generate a private identity and shareable public record")
    keys.add_argument("--member", required=True); keys.add_argument("--private", required=True)
    keys.add_argument("--public", required=True); keys.set_defaults(function=_keygen)
    doctor = sub.add_parser("doctor", help="Measure capabilities and optionally certify actual training")
    doctor.add_argument("--directory", default="."); doctor.add_argument("--device", default="auto")
    doctor.add_argument("--available-hours", type=float, default=0); doctor.add_argument("--output")
    doctor.add_argument("--certify", action="store_true"); doctor.add_argument("--peer-url")
    doctor.add_argument("--host-id", help="Operator-verified physical machine identity for admission checks")
    doctor.add_argument("--role", choices=["worker", "coordinator"])
    doctor.add_argument("--peer-member"); doctor.add_argument("--identity"); doctor.add_argument("--manifest")
    doctor.set_defaults(function=_doctor)
    for command in ("train", "resume"):
        train = sub.add_parser(command, help="Train local or explicitly configured remote synchronous stages")
        train.add_argument("--manifest", required=True); train.add_argument("--checkpoints")
        train.add_argument("--steps", type=int, default=3); train.add_argument("--device", default="cpu")
        train.add_argument("--base-path"); train.add_argument("--endpoints"); train.add_argument("--identity")
        train.add_argument("--timeout", type=float, default=300); train.add_argument("--mirror", action="append", default=[])
        train.add_argument("--max-blob-mib", type=int, default=1024)
        train.add_argument("--admission", help="Measured admission report required for models above 500M base parameters")
        train.set_defaults(function=_train)
    join = sub.add_parser("join", help="Serve one authenticated stage (loopback by default)")
    join.add_argument("--manifest", required=True); join.add_argument("--identity", required=True)
    join.add_argument("--stage", type=int, required=True); join.add_argument("--device", default="cpu")
    join.add_argument("--base-path"); join.add_argument("--host", default="127.0.0.1")
    join.add_argument("--port", type=int, default=49200); join.add_argument("--allow-external", action="store_true")
    join.add_argument("--dht", action="store_true"); join.add_argument("--initial-peer", action="append", default=[])
    join.add_argument("--announce-url"); join.set_defaults(function=_join)
    join.add_argument("--max-blob-mib", type=int, default=1024)
    export = sub.add_parser("export", help="Export adapter weights with pinned provenance")
    export.add_argument("--manifest", required=True); export.add_argument("--checkpoints")
    export.add_argument("--base-path", help="Override the local tokenizer/base location for a portable full-model export")
    export.add_argument("--output", required=True); export.set_defaults(function=_export)
    prepare = sub.add_parser("prepare-data", help="Audit, deduplicate and split a provenance-bearing JSONL corpus")
    prepare.add_argument("--input", required=True); prepare.add_argument("--output", required=True)
    prepare.add_argument("--objective", choices=["sft", "cpt"], default="sft")
    prepare.set_defaults(function=_prepare_data)
    experiments = sub.add_parser("experiments", help="List bounded scratch candidates or rank completed equal-budget results")
    experiments.add_argument("--select"); experiments.add_argument("--output"); experiments.set_defaults(function=_experiments)
    tokenizer = sub.add_parser("tokenizer", help="Train an independent scratch tokenizer on frozen local training data")
    tokenizer.add_argument("--input", nargs="+", required=True); tokenizer.add_argument("--output", required=True)
    tokenizer.add_argument("--provenance", required=True); tokenizer.add_argument("--vocab-size", type=int, choices=[512, 64000, 128000], default=64000)
    tokenizer.add_argument("--input-format", choices=["text", "sft_jsonl"], default="text")
    tokenizer.add_argument("--smoke-test", action="store_true"); tokenizer.set_defaults(function=_tokenizer)
    capacity = sub.add_parser("capacity", help="Count actual meta-model stages and test measured resource lower bounds")
    capacity.add_argument("--manifest", required=True); capacity.add_argument("--reports", nargs="*")
    capacity.add_argument("--coordinator"); capacity.add_argument("--observations"); capacity.add_argument("--target-tokens", type=int)
    capacity.add_argument("--output"); capacity.set_defaults(function=_capacity)
    evaluate = sub.add_parser("evaluate", help="Summarize fixed bilingual results; optionally execute in an immutable Docker image")
    evaluate.add_argument("--results", required=True); evaluate.add_argument("--baseline"); evaluate.add_argument("--output")
    evaluate.add_argument("--execute", action="store_true"); evaluate.add_argument("--image"); evaluate.set_defaults(function=_evaluate)
    for command in ("infer", "score"):
        inference = sub.add_parser(command, help="Evaluate a committed checkpoint without mutating weights")
        inference.add_argument("--manifest", required=True); inference.add_argument("--checkpoints")
        inference.add_argument("--device", default="cpu"); inference.add_argument("--base-path")
        inference.add_argument("--endpoints"); inference.add_argument("--identity"); inference.add_argument("--timeout", type=float, default=300)
        inference.add_argument("--max-blob-mib", type=int, default=1024)
        if command == "infer":
            inference.add_argument("--prompt", required=True); inference.add_argument("--max-new-tokens", type=int, default=32)
            inference.set_defaults(function=_infer)
        else:
            inference.add_argument("--split", choices=["validation", "test"], default="validation")
            inference.add_argument("--output"); inference.set_defaults(function=_score)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.threads < 1:
            raise ValueError("--threads must be positive")
        import torch
        torch.set_num_threads(args.threads)
        if getattr(args, "steps", 1) < 1:
            raise ValueError("--steps must be positive")
        args.function(args)
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, KeyError) as error:
        print(f"sakura: {error}", file=sys.stderr, flush=True)
        return 2
