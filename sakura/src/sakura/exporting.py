"""Text-only Hugging Face model export from a complete committed stage set."""
from __future__ import annotations

import json
from pathlib import Path


def export_full_checkpoint(stage_states: dict, config: dict, num_stages: int, output_directory):
    from safetensors.torch import save_file
    from .model import _text_dict, _hf
    if set(stage_states) != {str(i) for i in range(num_stages)}:
        raise ValueError("Full export requires every stage")
    target = Path(output_directory)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("Full-model export output must be empty")
    target.mkdir(parents=True, exist_ok=True)
    text = _text_dict(config)
    weight_map, total_size = {}, 0
    for stage_index in range(num_stages):
        snapshot = stage_states[str(stage_index)]
        if snapshot.get("mode") != "full" or snapshot.get("stage_index") != stage_index:
            raise ValueError("Expected a full-weight checkpoint with matching stage identity")
        start = text["num_hidden_layers"] * stage_index // num_stages
        state = {}
        filename = f"model-{stage_index + 1:05d}-of-{num_stages:05d}.safetensors"
        for key, tensor in snapshot["model"].items():
            if key.startswith("layers."):
                _, local_layer, suffix = key.split(".", 2)
                key = f"model.layers.{start + int(local_layer)}.{suffix}"
            elif key.startswith(("embed_tokens.", "norm.")):
                key = "model." + key
            if key in weight_map or "lora_" in key or ".base_layer." in key:
                raise ValueError("Duplicate/non-full model tensor in export")
            state[key] = tensor.detach().cpu().contiguous()
            weight_map[key] = filename
            total_size += tensor.numel() * tensor.element_size()
        save_file(state, str(target / filename), metadata={"format": "pt"})
    _, Config, _ = _hf()
    hf_config = Config(**text)
    hf_config.architectures = ["Qwen3_5MoeForCausalLM"]
    hf_config.dtype = "float32"
    hf_config.save_pretrained(str(target))
    (target / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2) + "\n", encoding="utf-8")
    return target
