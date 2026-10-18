"""Versioned run identity shared by trainers and workers."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def validate_manifest(value: dict) -> dict:
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported run manifest schema_version (expected 1)")
    if not isinstance(value.get("run_id"), str) or not value["run_id"].strip():
        raise ValueError("run_id is required")
    if value.get("mode") not in {"lora", "full"}:
        raise ValueError("mode must be lora or full; modes never share an optimizer round")
    model = value.get("model", {})
    if not isinstance(model.get("config"), dict) or not model.get("revision"):
        raise ValueError("model.config and immutable model.revision are required")
    config = model["config"].get("text_config", model["config"])
    stages = value.get("num_stages")
    if type(stages) is not int or not 1 <= stages <= config.get("num_hidden_layers", 0):
        raise ValueError("num_stages must be between 1 and the number of decoder layers")
    if config.get("use_cache", False):
        raise ValueError("Training manifests must set use_cache=false")
    if value.get("tokenizer", {}).get("kind") not in {"byte", "huggingface"}:
        raise ValueError("tokenizer.kind must be byte (smoke only) or huggingface")
    if not value.get("dataset", {}).get("revision"):
        raise ValueError("dataset.revision is required")
    if value["dataset"].get("objective", "sft") not in {"sft", "cpt"}:
        raise ValueError("dataset.objective must be sft or cpt")
    training = value.get("training", {})
    for key in ("sequence_length", "checkpoint_every"):
        if type(training.get(key)) is not int or training[key] <= 0:
            raise ValueError(f"training.{key} must be a positive integer")
    for key in ("learning_rate", "max_grad_norm"):
        v = training.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"training.{key} must be positive and finite")
    if not isinstance(value.get("members", {}), dict):
        raise ValueError("members must map member IDs to Ed25519 public keys")
    # A 35B run must be pinned to actual weights, never silently random-initialized.
    if model.get("pretrained_path"):
        if not re.fullmatch(r"[0-9a-f]{40}", str(model["revision"])):
            raise ValueError("Pretrained runs require the 40-character upstream commit revision")
        if value["tokenizer"].get("revision") != model["revision"]:
            raise ValueError("Pretrained model/tokenizer revisions must agree")
        if not model.get("artifacts") or not value["tokenizer"].get("artifacts"):
            raise ValueError("Pretrained runs require model and tokenizer content-hash inventories")
    canonical_json(value)  # rejects non-finite or unserializable values
    return value


def run_hash(manifest: dict) -> str:
    return hashlib.sha256(canonical_json(validate_manifest(manifest))).hexdigest()


def load_manifest(path: str | Path) -> dict:
    return validate_manifest(json.loads(Path(path).read_text(encoding="utf-8")))


def save_manifest(path: str | Path, manifest: dict) -> None:
    validate_manifest(manifest)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
