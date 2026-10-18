"""Frozen Qwen3.5 MoE pipeline inference with owned hybrid caches.

Prefill sends the prompt once. Every subsequent stage RPC processes exactly one
new token/activation; DeltaNet convolution/recurrent states and attention KV stay
on their owner. This avoids full-prefix recomputation. It does not guarantee a
WAN split is faster than a model that fits on one GPU.

Wrap ``CachedStageWorker.handle`` in the existing signed ``RpcServer``. A local
``LocalTransport`` is useful for testing and colocated stages; authentication and
TLS/private-network policy remain the transport's responsibility.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
from typing import Sequence
import uuid

import torch
from torch import Tensor, nn

from .doctor import resolve_device
from .manifest import run_hash
from .model import QwenStage, LoRALinear, build_stage, inject_lora, load_exported_lora, merge_lora


_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_INTEGER_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}


def _adapter_fingerprint(path) -> str:
    """Bind every owner to the same complete adapter without loading its tensors."""
    fingerprints = {}
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        digest, size = hashlib.sha256(), 0
        with (Path(path) / name).open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
                size += len(block)
        fingerprints[name] = {"bytes": size, "sha256": digest.hexdigest()}
    return hashlib.sha256(json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def resolve_inference_dtype(dtype: str | torch.dtype, device) -> torch.dtype:
    """CPU auto=FP32, CUDA auto=BF16 where supported else FP16, MPS auto=FP16."""
    selected = torch.device(device)
    if dtype == "auto":
        if selected.type == "cpu":
            return torch.float32
        if selected.type == "mps":
            return torch.float16
        with torch.cuda.device(selected):
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    names = {"float32": torch.float32, "fp32": torch.float32,
             "float16": torch.float16, "fp16": torch.float16,
             "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
    result = names.get(dtype, dtype)
    if result not in {torch.float32, torch.float16, torch.bfloat16}:
        raise ValueError("Inference dtype must be auto, float32, float16 or bfloat16")
    if selected.type == "cuda" and result == torch.bfloat16:
        with torch.cuda.device(selected):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("Requested CUDA device does not support BF16")
    return result


def _cache_bytes(cache) -> int:
    seen, total = set(), 0
    def count(value):
        nonlocal total
        if isinstance(value, Tensor):
            if not value.numel():
                return
            storage = value.untyped_storage()
            identity = (str(value.device), storage.data_ptr())
            if identity not in seen:
                seen.add(identity)
                total += storage.nbytes()
        elif isinstance(value, (tuple, list)):
            for item in value:
                count(item)
    for layer in cache.layers:
        for name in ("keys", "values", "conv_states", "recurrent_states"):
            count(getattr(layer, name, None))
    return total


class CachedInferenceStage(nn.Module):
    """Consume one QwenStage for inference; local cache indices own no remote layers.

    The supplied stage must not subsequently be shared with another cached model:
    its decoder attention/cache layer indices are localized in place. Parameters
    are never cloned, LoRA is merged, and no optimizer is constructed.
    """
    def __init__(self, stage: QwenStage, device="cpu", dtype: str | torch.dtype = "auto"):
        super().__init__()
        self.device = resolve_device(str(device))
        self.dtype = resolve_inference_dtype(dtype, self.device)
        self.stage = merge_lora(stage, inplace=True)
        self.stage.requires_grad_(False)
        for parameter in self.stage.parameters():
            parameter.grad = None
        self.stage.to(device=self.device, dtype=self.dtype)
        # Rotary frequencies are arithmetic constants, independent of weight dtype.
        self.stage.rotary_emb.float()
        self.stage.eval()
        start, end = stage.layer_range
        self.config = copy.deepcopy(stage.config)
        self.config.layer_types = list(stage.config.layer_types[start:end])
        self.config.num_hidden_layers = end - start
        self.config.use_cache, self.config.output_router_logits = True, False
        self.config._attn_implementation = "eager"
        for index, layer in enumerate(self.stage.layers):
            if hasattr(layer, "linear_attn"):
                layer.linear_attn.layer_idx = index
            if hasattr(layer, "self_attn"):
                layer.self_attn.layer_idx = index
        self.eval()

    def new_cache(self):
        from transformers.cache_utils import DynamicCache
        return DynamicCache(config=self.config)

    def estimated_cache_bytes(self, tokens: int) -> int:
        """Conservative retained-state estimate for batch one (not activation peak)."""
        config, element = self.config, torch.empty((), dtype=self.dtype).element_size()
        total = 0
        for kind in config.layer_types:
            if kind == "linear_attention":
                recurrent = config.linear_num_value_heads * config.linear_key_head_dim * config.linear_value_head_dim * 4
                convolution = (2 * config.linear_num_key_heads * config.linear_key_head_dim
                               + config.linear_num_value_heads * config.linear_value_head_dim)
                total += recurrent + convolution * config.linear_conv_kernel_dim * element
            else:
                total += 2 * config.num_key_value_heads * config.head_dim * tokens * element
        return total

    @torch.inference_mode()
    def forward(self, inputs: Tensor, cache, position: int) -> dict:
        from transformers.masking_utils import create_causal_mask
        hidden = self.stage.embed_tokens(inputs.to(self.device)) if self.stage.is_head else inputs.to(
            device=self.device, dtype=self.dtype)
        positions = torch.arange(position, position + hidden.shape[1], device=self.device)[None, :]
        rotary = self.stage.rotary_emb(hidden, positions)
        # Pure DeltaNet stages have no sequence-length-tracking attention cache.
        # Do not call HF mask helpers on such a cache: no attention mask is needed.
        causal = None
        if "full_attention" in self.config.layer_types:
            causal = create_causal_mask(config=self.config, inputs_embeds=hidden, attention_mask=None,
                                        past_key_values=cache, position_ids=positions)
        for kind, layer in zip(self.config.layer_types, self.stage.layers):
            hidden = layer(hidden, position_embeddings=rotary, position_ids=positions,
                           attention_mask=causal if kind == "full_attention" else None,
                           past_key_values=cache, use_cache=True)
        if self.stage.is_tail:
            # Avoid allocating [prompt_length,vocabulary] logits during prefill.
            last_hidden = self.stage.norm(hidden[:, -1:, :])
            next_token = self.stage.lm_head(last_hidden).argmax(-1).reshape(1)
            return {"next_token": next_token.cpu()}
        return {"hidden_states": hidden.contiguous().cpu()}


@dataclass
class _Session:
    cache: object
    position: int
    expires_at: float
    prefill_calls: int = 0
    decode_calls: int = 0
    processed_tokens: int = 0


class CachedStageWorker:
    """CachedStageWorker(manifest,stage_index,device='cpu',dtype='auto',*,
    base_path=None,adapter_path=None,stage=None,max_sessions=4,session_ttl=300,
    max_context=4096,max_cache_bytes=536870912).

    ``handle`` supports ``info``, ``reset``, ``prefill`` and ``decode``. A model
    session is unpadded batch one. Prefill request has position=0 and inputs token
    IDs (head) or prompt activations (body). Decode has exactly one new token/
    activation and position equal to the worker's current cache length. Tail
    returns next_token only; other stages return hidden_states. No training API,
    logits, router tensors, model weights, or optimizer state is exposed by RPC.
    ``adapter_path`` is an exported PEFT-format adapter directory.
    """
    def __init__(self, manifest: dict, stage_index: int, device="cpu", dtype="auto", *,
                 base_path=None, adapter_path=None, stage: QwenStage | None = None,
                 max_sessions: int = 4, session_ttl: float = 300,
                 max_context: int = 4096, max_cache_bytes: int = 536870912):
        self.manifest, self.run_hash = copy.deepcopy(manifest), run_hash(manifest)
        if type(stage_index) is not int or not 0 <= stage_index < manifest["num_stages"]:
            raise ValueError("Inference stage index out of range")
        if type(max_sessions) is not int or not 1 <= max_sessions <= 64:
            raise ValueError("max_sessions must be between 1 and 64")
        if (isinstance(session_ttl, bool) or not isinstance(session_ttl, (int, float))
                or not math.isfinite(session_ttl) or not 0 < session_ttl <= 86400):
            raise ValueError("session_ttl must be finite and between 0 and 86400 seconds")
        if type(max_context) is not int or max_context <= 0:
            raise ValueError("max_context must be a positive integer")
        if type(max_cache_bytes) is not int or max_cache_bytes <= 0:
            raise ValueError("max_cache_bytes must be a positive integer")
        selected = resolve_device(str(device))
        chosen_dtype = resolve_inference_dtype(dtype, selected)
        source = manifest["model"]
        if stage is None:
            local_path = base_path or source.get("pretrained_path")
            if local_path:
                from .artifacts import verify_snapshot
                verify_snapshot(local_path, source["artifacts"])
            stage = build_stage(source["config"], stage_index, manifest["num_stages"], mode="full",
                                pretrained_path=local_path, revision=source.get("revision"),
                                seed=source.get("seed", 0), device=str(selected), dtype=chosen_dtype)
        if (stage.stage_index, stage.num_stages) != (stage_index, manifest["num_stages"]):
            raise ValueError("Supplied stage ownership differs from manifest")
        self.adapter_sha256 = None
        if adapter_path is not None:
            self.adapter_sha256 = _adapter_fingerprint(adapter_path)
            if not any(isinstance(module, LoRALinear) for module in stage.modules()):
                options = json.loads((Path(adapter_path) / "adapter_config.json").read_text(encoding="utf-8"))
                inject_lora(stage, rank=options["r"], alpha=options["lora_alpha"],
                            dropout=options["lora_dropout"])
            load_exported_lora(stage, adapter_path)
            if _adapter_fingerprint(adapter_path) != self.adapter_sha256:
                raise ValueError("Adapter files changed while loading the inference stage")
        self.model = CachedInferenceStage(stage, selected, chosen_dtype)
        self.stage = self.model.stage
        self.stage_index, self.num_stages = stage_index, manifest["num_stages"]
        self.device, self.dtype = selected, chosen_dtype
        self.max_context = min(max_context, self.model.config.max_position_embeddings)
        self.max_sessions, self.session_ttl = max_sessions, float(session_ttl)
        self.max_cache_bytes = max_cache_bytes
        self.sessions: dict[str, _Session] = {}
        self.lock, self._closed = threading.RLock(), False

    def _expire(self):
        now = time.monotonic()
        for identifier, session in list(self.sessions.items()):
            if session.expires_at <= now:
                del self.sessions[identifier]

    def _identifier(self, request):
        identifier = request.get("session_id")
        if not isinstance(identifier, str) or not _SESSION_ID.fullmatch(identifier):
            raise ValueError("session_id must be 1..128 ASCII letters/digits/underscore/hyphen")
        return identifier

    def _inputs(self, value, operation: str) -> Tensor:
        if not isinstance(value, Tensor):
            raise TypeError("Inference inputs must be tensors")
        if self.stage.is_head:
            if value.ndim != 2 or value.shape[0] != 1 or value.dtype not in _INTEGER_DTYPES:
                raise ValueError("Head inputs must be batch-one integer token IDs [1,sequence]")
            if value.numel() and (int(value.min()) < 0 or int(value.max()) >= self.model.config.vocab_size):
                raise ValueError("Token ID outside model vocabulary")
        else:
            if (value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != self.model.config.hidden_size
                    or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
                raise ValueError("Body inputs must be finite batch-one activations [1,sequence,hidden]")
        if value.shape[1] < 1 or (operation == "decode" and value.shape[1] != 1):
            raise ValueError("Prefill must be nonempty; decode must contain exactly one new token")
        return value.to(dtype=torch.long) if self.stage.is_head else value

    def handle(self, request: dict) -> dict:
        with self.lock:
            if self._closed:
                raise RuntimeError("Inference worker is closed")
            self._expire()
            operation = request.get("op")
            if operation == "info":
                return {"run_hash": self.run_hash, "stage_index": self.stage_index,
                        "num_stages": self.num_stages, "inference": True,
                        "device": str(self.device), "dtype": str(self.dtype).removeprefix("torch."),
                        "cache_type": "hf-dynamic-hybrid-kv-deltanet", "max_context": self.max_context,
                        "max_sessions": self.max_sessions, "sessions": len(self.sessions),
                        "session_ttl": self.session_ttl, "max_cache_bytes": self.max_cache_bytes,
                        "cache_bytes": sum(_cache_bytes(session.cache) for session in self.sessions.values()),
                        "adapter_sha256": self.adapter_sha256,
                        "vocab_size": self.model.config.vocab_size, "hidden_size": self.model.config.hidden_size}
            if operation not in {"prefill", "decode", "reset"}:
                raise ValueError("Inference RPC supports info, prefill, decode and reset only")
            identifier = self._identifier(request)
            if operation == "reset":
                return {"reset": self.sessions.pop(identifier, None) is not None}
            position = request.get("position")
            if type(position) is not int or position < 0:
                raise ValueError("Inference request position must be a nonnegative integer")
            inputs = self._inputs(request.get("inputs"), operation)
            if operation == "prefill":
                if position != 0 or identifier in self.sessions:
                    raise ValueError("Prefill requires a new session at position=0; reset before reusing it")
                if len(self.sessions) >= self.max_sessions:
                    raise RuntimeError("Inference session capacity reached; reset a session or wait for TTL expiry")
                session = _Session(self.model.new_cache(), 0, time.monotonic() + self.session_ttl)
            else:
                session = self.sessions.get(identifier)
                if session is None:
                    raise ValueError("Unknown or expired inference session; prefill again")
                if session.position != position:
                    raise ValueError("Stale/out-of-order decode position; reset and prefill after an uncertain RPC")
            next_position = position + inputs.shape[1]
            if next_position > self.max_context:
                raise ValueError("Inference context limit reached")
            other_bytes = sum(_cache_bytes(other.cache) for key, other in self.sessions.items() if key != identifier)
            estimated = self.model.estimated_cache_bytes(next_position)
            if estimated + other_bytes > self.max_cache_bytes:
                raise MemoryError("Inference retained-cache byte budget exceeded")
            try:
                output = self.model(inputs, session.cache, position)
                actual_bytes = _cache_bytes(session.cache)
                if actual_bytes + other_bytes > self.max_cache_bytes:
                    raise MemoryError("Actual inference cache exceeds configured byte budget")
            except BaseException:
                # An interrupted layer may have updated only part of the cache.
                self.sessions.pop(identifier, None)
                raise
            session.position = next_position
            session.expires_at = time.monotonic() + self.session_ttl
            session.processed_tokens += inputs.shape[1]
            session.prefill_calls += operation == "prefill"
            session.decode_calls += operation == "decode"
            self.sessions[identifier] = session
            output.update(position=next_position, processed_tokens=inputs.shape[1], cache_bytes=actual_bytes)
            return output

    def close(self):
        with self.lock:
            self.sessions.clear()
            self._closed = True


class CachedPipeline:
    """CachedPipeline(endpoints,*,max_context=4096,expected_run_hash=None).

    Endpoints expose ``stage_index`` and ``call(request)`` (the existing
    ``StageEndpoint`` contract). ``generate(input_ids,max_new_tokens,
    eos_token_id=None,session_id=None)`` is greedy, unpadded batch one and returns
    CPU ``tokens`` [1,new_tokens], ``output_ids`` [1,prompt+new_tokens], and metrics.
    Session caches are reset at completion/failure; TTL bounds failed cleanup.
    Context limits count processed/cache tokens, including all but the last
    predicted token, which does not need another decoder invocation.
    """
    def __init__(self, endpoints: Sequence, *, max_context: int = 4096, expected_run_hash: str | None = None):
        if type(max_context) is not int or max_context < 1:
            raise ValueError("max_context must be a positive integer")
        self.endpoints = sorted(endpoints, key=lambda endpoint: endpoint.stage_index)
        if not self.endpoints or [endpoint.stage_index for endpoint in self.endpoints] != list(range(len(self.endpoints))):
            raise ValueError("Exactly one ordered endpoint per inference stage is required")
        infos = [endpoint.call({"op": "info"}) for endpoint in self.endpoints]
        identity = expected_run_hash or infos[0].get("run_hash")
        for index, info in enumerate(infos):
            if (not info.get("inference") or info.get("stage_index") != index
                    or info.get("num_stages") != len(infos) or info.get("run_hash") != identity):
                raise ValueError("Inference endpoint run/stage identity mismatch")
        if len({info["vocab_size"] for info in infos}) != 1 or len({info["hidden_size"] for info in infos}) != 1:
            raise ValueError("Inference stage model schemas differ")
        if len({info.get("adapter_sha256") for info in infos}) != 1:
            raise ValueError("Inference stages loaded different adapters")
        self.run_hash, self.vocab_size = identity, infos[0]["vocab_size"]
        self.max_context = min(max_context, *(info["max_context"] for info in infos))

    def generate(self, input_ids, max_new_tokens: int, eos_token_id=None, session_id: str | None = None) -> dict:
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        ids = torch.as_tensor(input_ids)
        if ids.dtype not in _INTEGER_DTYPES:
            raise ValueError("input_ids must contain integer tokens")
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
            raise ValueError("Generation requires one nonempty unpadded prompt")
        ids = ids.detach().to(device="cpu", dtype=torch.long).contiguous()
        if int(ids.min()) < 0 or int(ids.max()) >= self.vocab_size:
            raise ValueError("Token ID outside model vocabulary")
        if ids.shape[1] + max(max_new_tokens - 1, 0) > self.max_context:
            raise ValueError("Requested generation exceeds the pipeline cache/context limit")
        stops = set() if eos_token_id is None else ({eos_token_id} if type(eos_token_id) is int else set(eos_token_id))
        if any(type(token) is not int or not 0 <= token < self.vocab_size for token in stops):
            raise ValueError("EOS tokens must be valid integer vocabulary IDs")
        identifier = session_id or uuid.uuid4().hex
        if not isinstance(identifier, str) or not _SESSION_ID.fullmatch(identifier):
            raise ValueError("Invalid inference session_id")
        generated, position, wire_bytes = [], 0, 0
        prefill_seconds = decode_seconds = 0.0
        start = time.perf_counter()
        cleanup_failures = 0
        try:
            for token_index in range(max_new_tokens):
                step_started = time.perf_counter()
                inputs = ids if token_index == 0 else torch.tensor([[generated[-1]]], dtype=torch.long)
                operation = "prefill" if token_index == 0 else "decode"
                expected_position = position + inputs.shape[1]
                for endpoint in self.endpoints:
                    if inputs.is_floating_point():
                        wire_bytes += inputs.numel() * inputs.element_size()
                    reply = endpoint.call({"op": operation, "session_id": identifier, "position": position,
                                           "inputs": inputs})
                    if reply.get("position") != expected_position:
                        raise RuntimeError("Inference stage cache position mismatch")
                    if endpoint.stage_index < len(self.endpoints) - 1:
                        inputs = reply["hidden_states"]
                        wire_bytes += inputs.numel() * inputs.element_size()
                    else:
                        token = reply["next_token"]
                        if not isinstance(token, Tensor) or token.shape != (1,) or token.dtype not in _INTEGER_DTYPES:
                            raise RuntimeError("Tail returned an invalid next-token schema")
                        predicted = int(token.item())
                        if not 0 <= predicted < self.vocab_size:
                            raise RuntimeError("Tail returned an out-of-vocabulary token")
                position = expected_position
                elapsed = time.perf_counter() - step_started
                if token_index == 0:
                    prefill_seconds = elapsed
                else:
                    decode_seconds += elapsed
                generated.append(predicted)
                if predicted in stops:
                    break
        finally:
            if max_new_tokens:
                for endpoint in self.endpoints:
                    try:
                        endpoint.call({"op": "reset", "session_id": identifier})
                    except Exception:
                        cleanup_failures += 1
        total = time.perf_counter() - start
        tokens = torch.tensor([generated], dtype=torch.long)
        count = len(generated)
        return {"tokens": tokens, "output_ids": torch.cat((ids, tokens), dim=1),
                "metrics": {"prompt_tokens": ids.shape[1], "generated_tokens": count,
                            "prefill_seconds": prefill_seconds, "first_token_seconds": prefill_seconds,
                            "decode_seconds": decode_seconds, "total_seconds": total,
                            "tokens_per_second": count / total if total else 0.0,
                            "decode_tokens_per_second": max(count - 1, 0) / decode_seconds if decode_seconds else 0.0,
                            "stage_count": len(self.endpoints), "activation_bytes_sent": wire_bytes,
                            "activation_bytes_are_logical_payload": True, "cleanup_failures": cleanup_failures,
                            "cache_type": "hf-dynamic-hybrid-kv-deltanet"}}


# Compatibility aliases for the public inference API.
InferenceWorker = CachedStageWorker
PipelineInference = CachedPipeline
