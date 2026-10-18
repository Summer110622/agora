"""Portable, text-only Qwen3.5 MoE training and owned-stage checkpoint loading.

This package intentionally does not import Agora's native runtime.  The model is
the actual Hugging Face implementation, including its Torch reference DeltaNet.
``build_stage`` loads only locally owned tensors from a *local* safetensors
snapshot; constructing a worker never materializes a full pretrained model.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


LORA_TARGETS = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
    "in_proj_b", "in_proj_a", "out_proj",
})


def _hf():
    # These optional packages can silently select vendor-specific implementations
    # even when attention is eager.  A portable worker must use an isolated env.
    for package in ("fla", "causal_conv1d"):
        try:
            found = importlib.util.find_spec(package)
        except (ValueError, ImportError):
            found = None
        if found is not None:
            raise RuntimeError(
                f"Portable profile requires an environment without {package}; "
                "use the isolated sakura environment."
            )
    try:
        from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe
    except ImportError as exc:
        raise ImportError(
            "Qwen3.5 MoE requires the pinned sakura Transformers/PyTorch "
            "environment. The system Python environment is not modified."
        ) from exc
    return Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig, modeling_qwen3_5_moe


def tiny_config(vocab_size: int = 248320, hidden_size: int = 256,
                num_hidden_layers: int = 4, **overrides) -> dict:
    """Small *real* hybrid Qwen3.5, with the official vocabulary by default.

    Tests may explicitly use a synthetic small vocabulary.  Such a model is not
    tokenizer-compatible with the published checkpoint and is never mislabeled
    as pretrained or as SakuraMoE-35B.
    """
    head_dim = max(8, min(32, hidden_size // 8))
    rope_half = head_dim // 8  # partial_rotary_factor=.25, then /2
    section = [rope_half // 3] * 3
    for i in range(rope_half % 3):
        section[i] += 1
    config = {
        "model_type": "qwen3_5_moe_text", "vocab_size": vocab_size,
        "hidden_size": hidden_size, "num_hidden_layers": num_hidden_layers,
        "layer_types": ["full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                        for i in range(num_hidden_layers)],
        "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": head_dim,
        "linear_num_key_heads": 4, "linear_num_value_heads": 8,
        "linear_key_head_dim": head_dim, "linear_value_head_dim": head_dim,
        "linear_conv_kernel_dim": 4, "moe_intermediate_size": max(8, hidden_size // 2),
        "shared_expert_intermediate_size": max(8, hidden_size // 2),
        "num_experts": 8, "num_experts_per_tok": 2,
        "max_position_embeddings": 2048, "attention_dropout": 0.0,
        "router_aux_loss_coef": 0.001, "output_router_logits": True,
        "tie_word_embeddings": False, "use_cache": False,
        "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000.0,
                            "partial_rotary_factor": 0.25,
                            "mrope_section": section, "mrope_interleaved": True},
    }
    config.update(overrides)
    return config


def _text_dict(config) -> dict:
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    data = copy.deepcopy(dict(config))
    data = data.get("text_config", data)
    for name in ("lora", "seed", "pretrained_path", "revision", "mode", "name", "model_id",
                 "sakura_allow_large_random"):
        data.pop(name, None)
    data["model_type"] = "qwen3_5_moe_text"
    data["architectures"] = ["Qwen3_5MoeForCausalLM"]
    data["use_cache"] = False
    data["output_router_logits"] = True
    return data


class LoRALinear(nn.Module):
    """Ordinary Linear LoRA; packed expert tensors are deliberately untouched."""
    def __init__(self, base_layer: nn.Linear, rank: int = 16, alpha: float = 32,
                 dropout: float = 0.0):
        super().__init__()
        if rank <= 0 or dropout != 0.0:
            raise ValueError("Portable LoRA requires positive rank and dropout=0")
        self.base_layer = base_layer
        self.rank, self.alpha = rank, float(alpha)
        self.scaling = self.alpha / self.rank
        self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        base_layer.requires_grad_(False)
        options = {"device": base_layer.weight.device, "dtype": base_layer.weight.dtype}
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, **options))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, **options))
        self.reset_lora_parameters()

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, inputs: Tensor) -> Tensor:
        result = self.base_layer(inputs)
        adapter = F.linear(F.linear(inputs.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return result + (adapter * self.scaling).to(result.dtype)

    def merged(self) -> nn.Linear:
        layer = self.base_layer
        with torch.no_grad():
            delta = self.lora_B.float() @ self.lora_A.float()
            layer.weight.add_((delta * self.scaling).to(layer.weight))
        return layer


def _stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def inject_lora(model: nn.Module, rank: int = 16, alpha: float = 32,
                dropout: float = 0.0, seed: int = 0) -> int:
    model.requires_grad_(False)
    targets = [(name, module) for name, module in model.named_modules()
               if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in LORA_TARGETS
               and ("self_attn." in name or "linear_attn." in name)]
    for name, module in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        canonical = model.global_name(name) if isinstance(model, QwenStage) else name
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_stable_seed(seed, canonical))
            setattr(parent, attr, LoRALinear(module, rank, alpha, dropout))
    if not targets:
        raise ValueError("No attention/DeltaNet LoRA targets found")
    return len(targets)


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {name: param.detach().cpu().clone() for name, param in model.named_parameters()
            if name.endswith(("lora_A", "lora_B"))}


def load_lora_state_dict(model: nn.Module, state: Mapping[str, Tensor]) -> None:
    expected = {name: param for name, param in model.named_parameters()
                if name.endswith(("lora_A", "lora_B"))}
    if set(state) != set(expected):
        raise ValueError(f"Adapter keys mismatch: missing={sorted(set(expected)-set(state))}, "
                         f"unexpected={sorted(set(state)-set(expected))}")
    for name, param in expected.items():
        if state[name].shape != param.shape:
            raise ValueError(f"Adapter shape mismatch: {name}")
    with torch.no_grad():
        for name, param in expected.items():
            param.copy_(state[name].to(param))


def merge_lora(model: nn.Module, inplace: bool = True) -> nn.Module:
    """Restore ordinary HF Linear/state keys for ``save_pretrained`` export."""
    if not inplace:
        model = copy.deepcopy(model)
    targets = [(name, module) for name, module in model.named_modules()
               if isinstance(module, LoRALinear)]
    for name, module in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, module.merged())
    return model


def export_lora_checkpoint(stage_states: Mapping[str, dict], config: dict, num_stages: int,
                           output_dir, base_model_name_or_path: str, revision: str | None):
    """Export committed worker snapshots as ordinary PEFT adapter files.

    No model or pretrained tensors are constructed. ``stage_states`` may contain
    worker snapshot dictionaries (with a ``model`` field) or raw state dicts.
    A complete set of attention/DeltaNet A/B tensors is required.
    """
    from safetensors.torch import save_file
    text = _text_dict(config)
    layers = text["num_hidden_layers"]
    if not 1 <= num_stages <= layers:
        raise ValueError("Invalid stage count")
    states = {str(index): state for index, state in stage_states.items()}
    if set(states) != {str(index) for index in range(num_stages)}:
        raise ValueError("Adapter export requires every committed stage")
    tensors: dict[str, Tensor] = {}
    for index in range(num_stages):
        snapshot = states[str(index)]
        if "stage_index" in snapshot and snapshot["stage_index"] != index:
            raise ValueError("Snapshot stage identity mismatch")
        state = snapshot.get("model", snapshot)
        start, end = layers * index // num_stages, layers * (index + 1) // num_stages
        for name, tensor in state.items():
            if not name.endswith(("lora_A", "lora_B")):
                continue
            if not name.startswith("layers."):
                raise ValueError(f"Unexpected adapter owner: {name}")
            _, layer_index, suffix = name.split(".", 2)
            if not 0 <= int(layer_index) < end - start:
                raise ValueError("Adapter local layer index is outside owned range")
            global_name = f"model.layers.{start + int(layer_index)}.{suffix}"
            key = "base_model.model." + global_name + ".weight"
            if key in tensors or tensor.ndim != 2 or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid/duplicate adapter tensor: {key}")
            tensors[key] = tensor.detach().cpu().contiguous().clone()
    expected = set()
    kinds = text.get("layer_types", ["full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                                     for i in range(layers)])
    for layer_index, kind in enumerate(kinds):
        owner, targets = ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")) if kind == "full_attention" else (
            "linear_attn", ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"))
        for target in targets:
            for factor in ("lora_A", "lora_B"):
                expected.add(f"base_model.model.model.layers.{layer_index}.{owner}.{target}.{factor}.weight")
    if set(tensors) != expected:
        raise ValueError(f"Incomplete adapter schema: missing={sorted(expected-set(tensors))[:4]}, "
                         f"unexpected={sorted(set(tensors)-expected)[:4]}")
    ranks = {tensor.shape[0] if name.endswith("lora_A.weight") else tensor.shape[1]
             for name, tensor in tensors.items()}
    if len(ranks) != 1:
        raise ValueError("All adapter factors must use the same rank")
    rank = ranks.pop()
    options = config.get("lora", {})
    if rank != options.get("rank", options.get("r", 16)):
        raise ValueError("Adapter rank differs from the run configuration")
    adapter_config = {
        "base_model_name_or_path": str(base_model_name_or_path), "revision": revision,
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "inference_mode": True,
        "r": rank, "lora_alpha": options.get("alpha", 32), "lora_dropout": 0.0,
        "bias": "none", "fan_in_fan_out": False, "target_modules": sorted(LORA_TARGETS),
        "init_lora_weights": True, "modules_to_save": None,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output / "adapter_model.safetensors"), metadata={"format": "pt"})
    (output / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2) + "\n", encoding="utf-8")
    metadata = {"schema_version": 1, "format": "peft-lora", "num_stages": num_stages,
                "text_config": text, "base_model_name_or_path": str(base_model_name_or_path),
                "revision": revision, "rank": rank, "alpha": adapter_config["lora_alpha"]}
    (output / "sakura_adapter_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output


def load_exported_lora(model: nn.Module, path) -> None:
    """Load the owned subset of an exported adapter into our ordinary wrappers."""
    from safetensors import safe_open
    path = Path(path)
    metadata = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    if metadata.get("peft_type") != "LORA" or metadata.get("lora_dropout") != 0:
        raise ValueError("Not a portable LoRA adapter")
    expected = {name: parameter for name, parameter in model.named_parameters()
                if name.endswith(("lora_A", "lora_B"))}
    state = {}
    with safe_open(str(path / "adapter_model.safetensors"), framework="pt", device="cpu") as reader:
        available = set(reader.keys())
        for name in expected:
            canonical = model.global_name(name) if isinstance(model, QwenStage) else name
            key = "base_model.model." + canonical + ".weight"
            if key not in available:
                raise ValueError(f"Missing exported adapter factor: {key}")
            state[name] = reader.get_tensor(key)
    for module in model.modules():
        if isinstance(module, LoRALinear) and (module.rank != metadata["r"] or module.alpha != metadata["lora_alpha"]):
            raise ValueError("Adapter rank/alpha differs from the built wrappers")
    load_lora_state_dict(model, state)


def _portable(model: nn.Module, hf_module) -> None:
    model.config._attn_implementation = "eager"
    model.config.use_cache = False
    model.config.output_router_logits = True
    model._use_kernels = False
    # HF 5.x releases have either instance methods or module-level Torch funcs.
    for module in model.modules():
        if hasattr(module, "chunk_gated_delta_rule"):
            module.chunk_gated_delta_rule = hf_module.torch_chunk_gated_delta_rule
            module.recurrent_gated_delta_rule = hf_module.torch_recurrent_gated_delta_rule
            module.causal_conv1d_fn = None
    model.train()


def build_model(config: dict, mode: str = "lora", pretrained_path=None,
                revision=None, seed: int = 0) -> nn.Module:
    """Construct actual HF text model.  Full pretrained loading is local-only.

    Use ``build_stage`` for distributed workers: this convenience function does
    materialize the full model and is intended for tiny/reference runs or export.
    ``revision`` is recorded by the caller; a supplied local snapshot is immutable
    input here and no network resolution is performed.
    """
    if mode not in {"lora", "full"}:
        raise ValueError("mode must be 'lora' or 'full'")
    if pretrained_path is not None and not Path(pretrained_path).is_dir():
        raise ValueError("pretrained_path must be an existing LOCAL HF snapshot directory")
    Model, Config, hf_module = _hf()
    text = Config(**_text_dict(config))
    text._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Model(text)
    _portable(model, hf_module)
    if pretrained_path is not None:
        _load_owned_tensors(model, pretrained_path)
    model.requires_grad_(mode == "full")
    if mode == "lora":
        options = dict(config.get("lora", {}))
        inject_lora(model, rank=options.get("rank", options.get("r", 16)),
                    alpha=options.get("alpha", 32), dropout=options.get("dropout", 0.0), seed=seed)
    return model


@dataclass
class StageOutput:
    hidden_states: Tensor
    router_logits: tuple[Tensor, ...]
    logits: Tensor | None = None


@dataclass
class TrainingOutput:
    logits: Tensor
    loss: Tensor | None
    aux_loss: Tensor
    router_logits: tuple[Tensor, ...]


class QwenStage(nn.Module):
    """One contiguous decoder range, plus embedding/head at pipeline ends.

    Incoming floating hidden states can be detached transport tensors with
    ``requires_grad_(True)``. The caller returns their gradient upstream after
    backpropagating both the downstream hidden gradient and router-logit gradient.
    No hidden tensor is detached by this module.
    """
    def __init__(self, model, stage_index: int, num_stages: int):
        super().__init__()
        count = len(model.model.layers)
        if not 1 <= num_stages <= count or not 0 <= stage_index < num_stages:
            raise ValueError("Require 1 <= num_stages <= layers and valid stage_index")
        if model.config.tie_word_embeddings and num_stages > 1:
            raise ValueError("Cross-stage tied embeddings need shared-gradient ownership; use untied embeddings")
        self.config = copy.deepcopy(model.config)
        self.stage_index, self.num_stages = stage_index, num_stages
        self.is_head, self.is_tail = stage_index == 0, stage_index == num_stages - 1
        start = count * stage_index // num_stages
        end = count * (stage_index + 1) // num_stages
        self.layer_range = (start, end)
        self.layers = nn.ModuleList(list(model.model.layers)[start:end])
        self.rotary_emb = copy.deepcopy(model.model.rotary_emb)
        if self.is_head:
            self.embed_tokens = model.model.embed_tokens
        if self.is_tail:
            self.norm, self.lm_head = model.model.norm, model.lm_head

    @property
    def device(self):
        return next(self.parameters()).device

    def global_name(self, local_name: str) -> str:
        if local_name.startswith("layers."):
            parts = local_name.split(".", 2)
            global_layer = f"model.layers.{self.layer_range[0] + int(parts[1])}"
            return global_layer + ("." + parts[2] if len(parts) == 3 else "")
        if local_name.startswith(("embed_tokens.", "norm.")):
            return "model." + local_name
        if local_name in {"embed_tokens", "norm", "rotary_emb", "layers"}:
            return "model." + local_name
        return local_name

    def global_parameter_names(self) -> dict[str, str]:
        return {name: self.global_name(name) for name, _ in self.named_parameters()}

    def forward(self, hidden_or_ids: Tensor, attention_mask: Tensor | None = None,
                position_ids: Tensor | None = None) -> StageOutput:
        from transformers.masking_utils import create_causal_mask
        hidden = self.embed_tokens(hidden_or_ids) if self.is_head else hidden_or_ids
        if not self.is_head and not hidden.is_floating_point():
            raise TypeError("Body/tail input must be floating hidden states")
        if hidden.ndim != 3:
            raise ValueError("Hidden states must have [batch, sequence, hidden] shape")
        if attention_mask is not None:
            attention_mask = attention_mask.to(hidden.device)
            if attention_mask.shape != hidden.shape[:2]:
                raise ValueError("attention_mask shape must match batch/sequence")
        if position_ids is None:
            position_ids = torch.arange(hidden.shape[1], device=hidden.device)[None, :].expand(hidden.shape[0], -1)
        else:
            position_ids = position_ids.to(hidden.device)
        if position_ids.ndim != 2:
            raise ValueError("Text-only stages require [batch, sequence] position_ids")
        causal_mask = create_causal_mask(config=self.config, inputs_embeds=hidden,
                                        attention_mask=attention_mask, past_key_values=None,
                                        position_ids=position_ids)
        linear_mask = attention_mask
        if attention_mask is not None and bool(torch.all(attention_mask == 1)):
            linear_mask = None
        positions = self.rotary_emb(hidden, position_ids)
        routers: list[Tensor] = []
        handles = [layer.mlp.gate.register_forward_hook(
            lambda _module, _args, output: routers.append(output[0])) for layer in self.layers]
        try:
            for offset, layer in enumerate(self.layers):
                kind = self.config.layer_types[self.layer_range[0] + offset]
                hidden = layer(hidden, position_embeddings=positions,
                               attention_mask=linear_mask if kind == "linear_attention" else causal_mask,
                               position_ids=position_ids, past_key_values=None, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        logits = None
        if self.is_tail:
            hidden = self.norm(hidden)
            logits = self.lm_head(hidden)
        return StageOutput(hidden, tuple(routers), logits)


def partition_model(model, num_stages: int) -> list[QwenStage]:
    return [QwenStage(model, index, num_stages) for index in range(num_stages)]


def _checkpoint_index(path: Path) -> dict[str, Path]:
    if not path.is_dir():
        raise ValueError("pretrained_path must be an existing LOCAL HF snapshot directory")
    index = path / "model.safetensors.index.json"
    if index.is_file():
        raw = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        mapped = {name: (path / filename).resolve() for name, filename in raw.items()}
        root = path.resolve()
        if any(not file.is_relative_to(root) for file in mapped.values()):
            raise ValueError("Checkpoint shard path escapes snapshot directory")
        return mapped
    files = list(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors checkpoint in {path}")
    from safetensors import safe_open
    mapped = {}
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name in mapped:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                mapped[name] = file
    return mapped


def _load_owned_tensors(model: nn.Module, pretrained_path) -> None:
    """Stream only named parameters belonging to this model/stage, one at a time."""
    from safetensors import safe_open
    index = _checkpoint_index(Path(pretrained_path))
    requests: dict[Path, list[tuple[str, Tensor]]] = {}
    missing = []
    for local_name, param in model.named_parameters():
        if local_name.endswith(("lora_A", "lora_B")):
            continue
        name = model.global_name(local_name) if isinstance(model, QwenStage) else local_name
        name = name.replace(".base_layer.", ".")
        candidates = [name]
        if name.startswith("model."):
            candidates.append("model.language_model." + name[len("model."):])
        key = next((key for key in candidates if key in index), None)
        if key is None:
            # HF save_pretrained can unpack 3-D expert parameters into individual
            # per-expert Linear weights. Copy those directly into owned views;
            # never concatenate another full expert tensor in host memory.
            if name.endswith((".experts.gate_up_proj", ".experts.down_proj")):
                prefix, kind = name.rsplit(".", 1)
                for expert in range(param.shape[0]):
                    parts = ("gate_proj", "up_proj") if kind == "gate_up_proj" else ("down_proj",)
                    for part_index, part in enumerate(parts):
                        source_name = f"{prefix}.{expert}.{part}.weight"
                        choices = [source_name, "model.language_model." + source_name[len("model."):]]
                        source_key = next((candidate for candidate in choices if candidate in index), None)
                        if source_key is None:
                            missing.append(source_name)
                            continue
                        target = param[expert]
                        if kind == "gate_up_proj":
                            intermediate = target.shape[0] // 2
                            target = target[part_index * intermediate:(part_index + 1) * intermediate]
                        requests.setdefault(index[source_key], []).append((source_key, target))
            else:
                missing.append(name)
        else:
            requests.setdefault(index[key], []).append((key, param))
    if missing:
        raise ValueError(f"Missing text checkpoint parameters ({len(missing)}): {missing[:8]}")
    for file, params in requests.items():
        with safe_open(str(file), framework="pt", device="cpu") as reader:
            for key, param in params:
                tensor = reader.get_tensor(key)
                if tensor.shape != param.shape:
                    raise ValueError(f"Checkpoint shape mismatch for {key}: {tensor.shape} != {param.shape}")
                with torch.no_grad():
                    param.copy_(tensor.to(device=param.device, dtype=param.dtype))
                del tensor


def _available_host_memory() -> int:
    import psutil
    return int(psutil.virtual_memory().available)


def _admit_owned_initialization(stage: QwenStage, mode: str, config: dict,
                               device, dtype: torch.dtype) -> dict:
    """Check parameter/optimizer memory floors before materializing meta tensors.

    This is admission, not a claim that activations fit. The worker's measured
    forward/backward certification remains mandatory for a real run shape.
    Initial random numbers are generated on CPU for backend-independent weights.
    """
    destination = torch.device(device)
    parameters = sum(parameter.numel() for parameter in stage.parameters())
    if mode == "full":
        trainable = parameters
    else:
        options = config.get("lora", {})
        rank = options.get("rank", options.get("r", 16))
        trainable = sum(rank * (module.in_features + module.out_features)
                        for name, module in stage.named_modules()
                        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in LORA_TARGETS
                        and ("self_attn." in name or "linear_attn." in name))
    element_bytes = torch.empty((), dtype=dtype).element_size()
    reserve = 64 * 1024 * 1024
    # FP32 CPU initialization; CPU optimizer master/gradient/two Adam moments.
    # For CPU execution the initial weights are the optimizer master themselves.
    cpu_initialization = 4 * parameters
    cpu_training = 4 * parameters + 12 * trainable if destination.type == "cpu" else 16 * trainable
    if destination.type == "mps":
        # MPS and CPU allocations share physical RAM. Include both GPU weights /
        # gradients and CPU optimizer copies, rather than budgeting them twice
        # against independent fictitious memory pools.
        cpu_training += element_bytes * (parameters + trainable)
    required_host = max(cpu_initialization, cpu_training) + reserve
    available_host = _available_host_memory()
    if required_host > int(available_host * 0.70):
        raise MemoryError(f"Owned stage CPU RAM admission rejected: requires at least {required_host} bytes "
                          f"plus 30% headroom; available={available_host}. No stage tensors were allocated.")
    budget = {"parameter_count": parameters, "trainable_parameter_count": trainable,
              "required_host_bytes": required_host, "available_host_bytes": available_host,
              "headroom_fraction": 0.30, "activation_reserve_bytes": reserve,
              "note": "Parameter/optimizer memory floor only; measured training certification is still required"}
    if destination.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested CUDA/ROCm device is unavailable")
        free, _ = torch.cuda.mem_get_info(destination)
        required_device = element_bytes * (parameters + trainable) + reserve
        if required_device > int(free * 0.70):
            raise MemoryError(f"Owned stage GPU memory admission rejected: requires at least {required_device} "
                              f"bytes plus 30% headroom; available={free}")
        budget.update(required_device_bytes=required_device, available_device_bytes=int(free))
    elif destination.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested MPS device is unavailable")
    elif destination.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("Portable stages support cpu, CUDA/ROCm and mps")
    return budget


def _initialize_owned_stage(stage: QwenStage, hf_module, seed: int) -> None:
    """Actual HF initialization, independently seeded by global module identity."""
    initializer = hf_module.Qwen3_5MoePreTrainedModel(stage.config)
    # Meta post_init marks parameters initialized even though no values exist.
    # Reset those flags before invoking the original HF initialization methods.
    for tensor in (*stage.parameters(), *stage.buffers()):
        tensor._is_hf_initialized = False
    for name, module in reversed(list(stage.named_modules())):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_stable_seed(seed, stage.global_name(name)))
            initializer._init_weights(module)
            # Pinned HF's uniform(0,16).log() can in principle draw exactly zero.
            # Keep that degenerate decay finite without changing ordinary draws.
            if isinstance(module, hf_module.Qwen3_5MoeGatedDeltaNet):
                with torch.no_grad():
                    module.A_log.nan_to_num_(neginf=math.log(torch.finfo(torch.float32).tiny))
    for tensor in (*stage.parameters(), *stage.buffers()):
        tensor._is_hf_initialized = True


def build_stage(config: dict, stage_index: int, num_stages: int, mode: str = "lora",
                pretrained_path=None, revision=None, seed: int = 0,
                device="cpu", dtype: torch.dtype = torch.float32) -> QwenStage:
    """Build a worker without ever allocating another stage's pretrained weights.

    A local pretrained checkpoint always uses meta construction, prune, then
    owned-tensor materialization. Ordinary tiny runs match ``build_model``'s
    initialization. Explicit ``sakura_allow_large_random=True`` instead uses meta
    construction and partition-independent owned initialization, including for
    tiny tests of the scratch path. Large random models remain blocked by default.
    """
    if mode not in {"lora", "full"}:
        raise ValueError("mode must be 'lora' or 'full'")
    if pretrained_path is not None and not Path(pretrained_path).is_dir():
        raise ValueError("pretrained_path must be an existing LOCAL HF snapshot directory")
    scratch_flag = config.get("sakura_allow_large_random", config.get("text_config", {}).get(
        "sakura_allow_large_random", False))
    if type(scratch_flag) is not bool:
        raise ValueError("sakura_allow_large_random must be an explicit boolean")
    if pretrained_path is None and not scratch_flag:
        text = _text_dict(config)
        rough_parameters = (2 * text.get("vocab_size", 248320) * text.get("hidden_size", 2048)
            + text.get("num_hidden_layers", 40) * text.get("num_experts", 256) * 3
            * text.get("hidden_size", 2048) * text.get("moe_intermediate_size", 512))
        if rough_parameters > 500_000_000:
            raise ValueError("Large random stages require explicit sakura_allow_large_random=True and memory admission")
        model = build_model(config, mode=mode, seed=seed)
        stage = QwenStage(model, stage_index, num_stages)
        del model
        return stage.to(device=device, dtype=dtype)
    Model, Config, hf_module = _hf()
    text_config = Config(**_text_dict(config))
    text_config._attn_implementation = "eager"
    with torch.device("meta"):
        meta_model = Model(text_config)
        stage = QwenStage(meta_model, stage_index, num_stages)
    del meta_model
    if pretrained_path is None:
        budget = _admit_owned_initialization(stage, mode, config, device, dtype)
        stage.to_empty(device="cpu")
        stage.rotary_emb = hf_module.Qwen3_5MoeTextRotaryEmbedding(stage.config)
        _initialize_owned_stage(stage, hf_module, seed)
        stage.to(device=device, dtype=dtype)
        # Rotary frequencies stay FP32 even if an optimized caller chooses BF16.
        stage.rotary_emb = hf_module.Qwen3_5MoeTextRotaryEmbedding(stage.config).to(device=device)
        stage.admission_budget = budget
    else:
        stage.to(dtype=dtype)
        stage.to_empty(device=device)
        # Rotary frequencies are nonpersistent computed buffers, not checkpoint tensors.
        stage.rotary_emb = hf_module.Qwen3_5MoeTextRotaryEmbedding(stage.config).to(device=device)
        _load_owned_tensors(stage, pretrained_path)
    _portable(stage, hf_module)
    stage.requires_grad_(mode == "full")
    if mode == "lora":
        options = dict(config.get("lora", {}))
        inject_lora(stage, rank=options.get("rank", options.get("r", 16)),
                    alpha=options.get("alpha", 32), dropout=options.get("dropout", 0), seed=seed)
    return stage


def router_aux_loss(router_logits: Sequence[Tensor], num_experts: int, top_k: int,
                    attention_mask: Tensor | None = None) -> Tensor:
    """Global HF auxiliary objective, with autograd across all stage outputs."""
    if not router_logits:
        raise ValueError("Router logits are required for MoE auxiliary loss")
    _, _, hf_module = _hf()
    return hf_module.load_balancing_loss_func(tuple(router_logits), num_experts, top_k, attention_mask)


def causal_lm_loss(logits: Tensor, labels: Tensor) -> Tensor:
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must match [batch, sequence]")
    labels = labels.to(logits.device)
    valid = labels[:, 1:] != -100
    if not bool(valid.any()):
        raise ValueError("Batch has no supervised next-token labels")
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                           labels[:, 1:].reshape(-1), ignore_index=-100)


def forward_stages(stages: Sequence[QwenStage], input_ids: Tensor,
                   attention_mask: Tensor | None = None, labels: Tensor | None = None,
                   position_ids: Tensor | None = None,
                   router_aux_loss_coef: float | None = None) -> TrainingOutput:
    if not stages or not stages[0].is_head or not stages[-1].is_tail:
        raise ValueError("A complete ordered head/body/tail pipeline is required")
    if [stage.stage_index for stage in stages] != list(range(len(stages))):
        raise ValueError("Stages must be in pipeline order")
    hidden = input_ids
    routers: list[Tensor] = []
    for stage in stages:
        output = stage(hidden.to(stage.device), attention_mask, position_ids)
        hidden = output.hidden_states
        routers.extend(output.router_logits)
    logits = output.logits
    config = stages[0].config
    aux = router_aux_loss(routers, config.num_experts, config.num_experts_per_tok, attention_mask)
    loss = None
    if labels is not None:
        coefficient = config.router_aux_loss_coef if router_aux_loss_coef is None else router_aux_loss_coef
        loss = causal_lm_loss(logits, labels) + coefficient * aux.to(logits.device)
    return TrainingOutput(logits, loss, aux, tuple(routers))
