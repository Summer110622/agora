"""Measured device capabilities; unavailable evidence is reported, never invented."""
from __future__ import annotations

import importlib.metadata
import platform
import shutil
import statistics
import time
from pathlib import Path

import psutil


def resolve_device(requested: str = "auto"):
    import torch
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "rocm":
        if not torch.version.hip:
            raise ValueError("This PyTorch build has no ROCm/HIP backend")
        requested = "cuda"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA/ROCm backend is unavailable; refusing silent CPU fallback")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested MPS backend is unavailable; refusing silent CPU fallback")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("Supported devices: cpu, cuda[:index], rocm, mps")
    return device


def probe(directory: str | Path = ".", device: str = "auto", available_hours: float = 0) -> dict:
    import torch
    if available_hours < 0:
        raise ValueError("available_hours must be non-negative")
    selected = resolve_device(device)
    memory = psutil.virtual_memory()
    result = {
        "schema_version": 1, "measured_at": time.time(),
        "platform": platform.platform(), "architecture": platform.machine(),
        "cpu": {"logical_cores": psutil.cpu_count(), "physical_cores": psutil.cpu_count(logical=False)},
        "memory": {"total_bytes": memory.total, "available_bytes": memory.available},
        "disk": {"path": str(Path(directory).resolve()), "free_bytes": shutil.disk_usage(directory).free},
        "device": str(selected), "backend": "rocm" if selected.type == "cuda" and torch.version.hip else selected.type,
        "available_hours": available_hours,
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "huggingface-hub")},
        "network": {"status": "unmeasured"}, "training_certification": {"status": "not_run"},
    }
    if selected.type == "cuda":
        free, total = torch.cuda.mem_get_info(selected)
        result["accelerator"] = {"name": torch.cuda.get_device_name(selected), "total_bytes": total,
                                 "free_bytes": free, "bf16": torch.cuda.is_bf16_supported()}
    elif selected.type == "mps":
        result["accelerator"] = {"name": "Apple MPS", "unified_memory": True,
                                 "recommended_bytes": torch.mps.recommended_max_memory()}
    return result


def measure_network(client, url: str, member: str, payload_bytes: int = 262144) -> dict:
    """Measure an authenticated echo to the actual worker, not an unrelated speedtest."""
    if not 1024 <= payload_bytes <= 1024 * 1024:
        raise ValueError("Probe payload must be between 1 KiB and 1 MiB")
    latencies, elapsed = [], []
    for _ in range(3):
        start = time.perf_counter()
        if client.call(url, {"op": "probe", "payload": "ok"}, expected_member=member)["payload"] != "ok":
            raise ValueError("Probe response mismatch")
        latencies.append(time.perf_counter() - start)
        payload = "x" * payload_bytes
        start = time.perf_counter()
        reply = client.call(url, {"op": "probe", "payload": payload}, expected_member=member)
        elapsed.append(time.perf_counter() - start)
        if reply["payload"] != payload:
            raise ValueError("Probe payload mismatch")
    return {"status": "measured", "peer": member, "latency_ms": statistics.median(latencies) * 1000,
            "roundtrip_payload_mbps": 2 * payload_bytes * 8 / statistics.median(elapsed) / 1e6,
            "payload_bytes_each_direction": payload_bytes,
            "note": "Authenticated round-trip payload throughput; not separate upload/download capacity"}


def place_stages(stage_bytes: list[int], reports: list[dict], margin: float = 0.30) -> dict:
    """Conservative host-memory admission (offload included), never active-parameter sizing."""
    if not 0 <= margin < 1 or not stage_bytes or any(x <= 0 for x in stage_bytes):
        raise ValueError("positive stage sizes and margin in [0,1) required")
    remaining = [int(r["memory"]["available_bytes"] * (1 - margin)) for r in reports]
    placements = []
    for stage, required in sorted(enumerate(stage_bytes), key=lambda item: -item[1]):
        candidates = [i for i, free in enumerate(remaining) if free >= required]
        if not candidates:
            return {"feasible": False, "reason": f"No host can hold stage {stage} ({required} bytes)",
                    "placements": [], "remaining_bytes": remaining}
        host = max(candidates, key=lambda i: remaining[i])
        remaining[host] -= required
        placements.append({"stage": stage, "host_index": host, "required_bytes": required})
    return {"feasible": True, "placements": sorted(placements, key=lambda p: p["stage"]),
            "remaining_bytes": remaining, "margin": margin,
            "note": "Host-memory admission only; accelerator peak and throughput require training certification"}
