"""Inference-only HF generation, without training optimizers or checkpoint RAM."""
from __future__ import annotations

import re
import statistics
import time

import torch

from .doctor import resolve_device
from .inference_gateway import validate_generation


DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def inference_dtype(device, value="auto"):
    device = torch.device(device)
    if value == "auto":
        if device.type == "cpu":
            return torch.float32
        if device.type == "cuda":
            with torch.cuda.device(device):
                return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float16
    choices = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if value not in choices:
        raise ValueError("dtype must be auto, float32, float16 or bfloat16")
    if device.type == "cpu" and value != "float32":
        raise ValueError("Portable CPU inference uses float32")
    if device.type == "mps" and value == "bfloat16":
        raise ValueError("Portable MPS profile uses float16 or float32")
    return choices[value]


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class HFGenerator:
    def __init__(self, model, tokenizer, *, max_context=4096, chat_template=True):
        if type(max_context) is not int or not 2 <= max_context <= 131072:
            raise ValueError("max_context must be in 2..131072")
        self.model, self.tokenizer = model.eval().requires_grad_(False), tokenizer
        self.device = next(model.parameters()).device
        configured = getattr(model.config, "text_config", model.config)
        self.max_context = min(max_context, getattr(configured, "max_position_embeddings", max_context))
        self.chat_template = chat_template

    @classmethod
    def from_pretrained(cls, model_id=DEFAULT_MODEL, revision=DEFAULT_REVISION, *, device="cpu",
                        dtype="auto", max_context=4096, chat_template=True):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Model and tokenizer require the same immutable 40-character commit SHA")
        from transformers import (AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer)
        selected = resolve_device(device)
        config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        # Qwen3.5 published repos wrap their text decoder with a vision module.
        # Select the actual architecture instead of coercing dense weights to MoE.
        loader = AutoModelForImageTextToText if hasattr(config, "vision_config") else AutoModelForCausalLM
        model = loader.from_pretrained(model_id, revision=revision, config=config, trust_remote_code=False,
                                       use_safetensors=True, dtype=inference_dtype(selected, dtype),
                                       attn_implementation="sdpa").to(selected)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        return cls(model, tokenizer, max_context=max_context, chat_template=chat_template)

    def _encode(self, prompt):
        if self.chat_template and getattr(self.tokenizer, "chat_template", None):
            ids = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                     tokenize=True, add_generation_prompt=True, return_dict=False)
        else:
            ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 64, *, use_cache=True) -> dict:
        validate_generation(prompt, max_new_tokens)
        ids = self._encode(prompt)
        if ids.shape[1] < 1 or ids.shape[1] + max_new_tokens > self.max_context:
            raise ValueError(f"Prompt plus generation exceeds context limit {self.max_context}; no silent truncation")
        _synchronize(self.device)
        start = time.perf_counter()
        output = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                     max_new_tokens=max_new_tokens, do_sample=False, use_cache=use_cache,
                                     pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                                                   else self.tokenizer.eos_token_id),
                                     logits_to_keep=1)
        _synchronize(self.device)
        elapsed = time.perf_counter() - start
        tokens = output[0, ids.shape[1]:].cpu().tolist()
        return {"text": self.tokenizer.decode(tokens, skip_special_tokens=True), "token_ids": tokens,
                "metrics": {"prompt_tokens": ids.shape[1], "generated_tokens": len(tokens),
                            "generation_seconds": elapsed, "tokens_per_second": len(tokens) / max(elapsed, 1e-9),
                            "use_cache": use_cache, "device": str(self.device),
                            "dtype": str(next(self.model.parameters()).dtype), "includes_model_load": False}}

    def benchmark(self, prompt, max_new_tokens=64, *, repeats=3):
        if type(repeats) is not int or not 1 <= repeats <= 10:
            raise ValueError("Benchmark repeats must be in 1..10")
        self.generate(prompt, max_new_tokens=min(max_new_tokens, 2))
        samples = {True: [], False: []}
        reference = None
        for iteration in range(repeats):
            # Alternate ordering to reduce warm-cache/order bias.
            for cached in ((True, False) if iteration % 2 == 0 else (False, True)):
                result = self.generate(prompt, max_new_tokens=max_new_tokens, use_cache=cached)
                if reference is None:
                    reference = result["token_ids"]
                if result["token_ids"] != reference:
                    raise RuntimeError("Cached and uncached greedy tokens differ; speed comparison is not certified")
                samples[cached].append(result["metrics"]["generation_seconds"])
        cached_time, uncached_time = (statistics.median(samples[v]) for v in (True, False))
        return {"cached_seconds": samples[True], "uncached_seconds": samples[False],
                "cached_median_seconds": cached_time, "uncached_median_seconds": uncached_time,
                "measured_cache_speedup": uncached_time / max(cached_time, 1e-9),
                "greedy_tokens_equal": True, "generated_tokens": len(reference),
                "scope": "same-device single-model warm generation; no WAN or cross-device speedup claim"}
