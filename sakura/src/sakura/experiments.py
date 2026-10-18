"""Reproducible architecture candidates and evidence-based scheduling.

Parameter counts are an explicit analytical proxy, not measured Qwen capacity or
proof that a 35B model fits. Candidate export never allocates a model or tokenizer.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


def estimate_capacity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Estimate a tied-embedding, SwiGLU MoE with configurable attention costs.

    Active means the routed structural parameter count including full embeddings,
    router and norms, not FLOPs or bytes touched per token. Full attention defaults
    to 4*h*h (dense MHA); hybrid defaults to the *same cost proxy* unless callers
    supply measured linear_attention_parameters_per_layer. Production architecture
    and GQA/DeltaNet counts must be supplied from the actual implementation.
    """
    names = ("hidden_size", "layers", "experts", "top_k", "expert_width", "vocab_size")
    values = {name: config.get(name) for name in names}
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values.values()):
        raise ValueError("capacity dimensions must be positive integers")
    hidden, layers, experts, top_k, width, vocab = (values[name] for name in names)
    if top_k > experts:
        raise ValueError("top_k cannot exceed experts")
    attention = config.get("attention", "full")
    if attention not in {"full", "hybrid"}:
        raise ValueError("attention must be full or hybrid")
    full_cost = config.get("full_attention_parameters_per_layer", 4 * hidden * hidden)
    linear_cost = config.get("linear_attention_parameters_per_layer", 4 * hidden * hidden)
    if any(not isinstance(cost, int) or isinstance(cost, bool) or cost <= 0 for cost in (full_cost, linear_cost)):
        raise ValueError("attention parameter costs must be positive integers")
    interval = config.get("full_attention_interval", 4)
    if not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0:
        raise ValueError("full_attention_interval must be a positive integer")
    full_layers = layers if attention == "full" else layers // interval
    linear_layers = layers - full_layers
    embeddings = vocab * hidden * (1 if config.get("tie_embeddings", True) else 2)
    attention_parameters = full_layers * full_cost + linear_layers * linear_cost
    router_and_norms = layers * (hidden * experts + 2 * hidden) + hidden
    per_expert = 3 * hidden * width
    shared = embeddings + attention_parameters + router_and_norms
    total = shared + layers * experts * per_expert
    active = shared + layers * top_k * per_expert
    return {"total_parameters": total, "active_parameters": active,
            "shared_parameters": shared, "expert_parameters": layers * experts * per_expert,
            "active_expert_parameters": layers * top_k * per_expert,
            "full_attention_layers": full_layers, "linear_attention_layers": linear_layers,
            "weight_bytes_fp32": total * 4, "weight_bytes_bf16": total * 2,
            "estimated": True,
            "attention_cost_is_proxy": attention == "hybrid" and "linear_attention_parameters_per_layer" not in config,
            "formula": "shared = embeddings + attention + router + norms; total = shared + L*E*3*h*w; active = shared + L*k*3*h*w",
            "active_definition": "routed structural parameters including full embedding tables; not compute or memory estimates"}


def experiment_grid(
    *, hidden_size: int = 128, layers: int = 4, base_expert_width: int = 128,
) -> list[dict[str, Any]]:
    """12 equal-expert-capacity candidates; 64k/128k tokenizer *targets* only.

    At 64 experts width=w; at 128 width=w/2; at 256 width=w/4.
    Thus E*w and k*w remain constant; tokenizer/attention shared costs are reported
    separately. Existing Qwen tokenizers must never be resized for this grid.
    """
    if not isinstance(base_expert_width, int) or isinstance(base_expert_width, bool) or base_expert_width <= 0 or base_expert_width % 4:
        raise ValueError("base_expert_width must be a positive multiple of four")
    candidates = []
    for attention in ("full", "hybrid"):
        for experts, top_k in ((64, 2), (128, 4), (256, 8)):
            for vocab in (64000, 128000):
                config = {"experiment_id": f"{attention}-e{experts}-k{top_k}-v{vocab}",
                          "attention": attention, "hidden_size": hidden_size, "layers": layers,
                          "experts": experts, "top_k": top_k, "expert_width": base_expert_width * 64 // experts,
                          "vocab_size": vocab, "tie_embeddings": True, "full_attention_interval": 4,
                          "tokenizer_status": "candidate target; train and version separately for scratch experiments"}
                config["capacity"] = estimate_capacity(config)
                candidates.append(config)
    return candidates


def candidate_to_model_config(
    candidate: Mapping[str, Any], *, proxy: bool = True, synthetic_vocab_size: int | None = None,
) -> dict[str, Any]:
    """Map a grid candidate to an executable, random-init Qwen3.5 text model.

    Proxy mode explicitly fixes hidden=64 and layers=4 while retaining all routed
    experts, top-k, inverse expert widths and the selected attention topology.
    A synthetic vocabulary is an opt-in smoke check, incompatible with the actual
    64k/128k candidate tokenizer. It never represents a chosen 35B architecture.
    Actual HF parameter counts must be measured from the constructed model.
    """
    estimate_capacity(candidate)  # validate dimensions before importing Torch
    if not isinstance(proxy, bool):
        raise ValueError("proxy must be boolean")
    if synthetic_vocab_size is not None and (not isinstance(synthetic_vocab_size, int)
            or isinstance(synthetic_vocab_size, bool) or synthetic_vocab_size < 8):
        raise ValueError("synthetic_vocab_size must be an explicit integer >= 8")
    from .model import tiny_config

    hidden = 64 if proxy else candidate["hidden_size"]
    layers = 4 if proxy else candidate["layers"]
    interval = candidate.get("full_attention_interval", 4)
    layer_types = ["full_attention" if candidate["attention"] == "full" or (i + 1) % interval == 0
                   else "linear_attention" for i in range(layers)]
    vocab = candidate["vocab_size"] if synthetic_vocab_size is None else synthetic_vocab_size
    # Untied tables are deliberate: the synchronous stage runtime rejects tied
    # cross-stage ownership until shared-gradient handling has been implemented.
    config = tiny_config(vocab_size=vocab, hidden_size=hidden, num_hidden_layers=layers,
                         layer_types=layer_types, num_experts=candidate["experts"],
                         num_experts_per_tok=candidate["top_k"],
                         moe_intermediate_size=candidate["expert_width"],
                         shared_expert_intermediate_size=max(8, hidden // 2),
                         tie_word_embeddings=False)
    config["_sakura_experiment"] = {
        "experiment_id": candidate.get("experiment_id"), "random_initialization": True,
        "proxy": proxy, "synthetic_vocabulary": synthetic_vocab_size is not None,
        "candidate_vocab_size": candidate["vocab_size"], "actual_vocab_size": vocab,
        "architecture_status": "runnable comparison infrastructure; no 35B architecture has been selected",
        "parameter_estimate_warning": "grid analytical counts omit Qwen shared-expert details and use different attention/table assumptions; measure this actual model",
    }
    return config


def train_tokenizer(
    corpus_paths: Iterable[str | Path], output_directory: str | Path, *,
    provenance: Iterable[Mapping[str, Any]], vocab_size: int = 64000,
    input_format: str = "text", smoke_test: bool = False, min_frequency: int = 2,
) -> dict[str, Any]:
    """Train an independent scratch ByteLevel BPE from local training data only.

    Official implementation:
    https://github.com/huggingface/tokenizers/blob/main/bindings/python/py_src/tokenizers/implementations/byte_level_bpe.py

    ``text`` streams UTF-8 lines; ``sft_jsonl`` streams validated SFT message
    content. Pass only the frozen training split; validation/test rows explicitly
    marked with a split are rejected. Provenance is mandatory. Inputs are hashed
    before/after streaming and never changed. No network operation is performed.
    Small vocabularies and insufficient corpora require smoke_test=True; ordinary
    runs must actually reach the requested 64k or 128k vocabulary.
    """
    from .data import SFTRecord, SourceProvenance

    if not isinstance(smoke_test, bool):
        raise ValueError("smoke_test must be boolean")
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size < 262:
        raise ValueError("ByteLevel BPE requires vocab_size >= 262 (256 bytes and six special tokens)")
    if not smoke_test and vocab_size not in {64000, 128000}:
        raise ValueError("scratch comparison vocab_size must be 64000 or 128000; small test vocabularies require smoke_test=True")
    if not isinstance(min_frequency, int) or isinstance(min_frequency, bool) or min_frequency <= 0:
        raise ValueError("min_frequency must be a positive integer")
    if input_format not in {"text", "sft_jsonl"}:
        raise ValueError("input_format must be text or sft_jsonl")
    sources = [SourceProvenance.from_dict(source) for source in provenance]
    if not sources:
        raise ValueError("tokenizer training requires source/license/revision provenance")
    paths = [Path(path).resolve() for path in corpus_paths]
    if not paths or any(not path.is_file() for path in paths) or len(set(paths)) != len(paths):
        raise ValueError("corpus_paths must name distinct existing local files")
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError("tokenizer output must be a new directory; original tokenizers are never overwritten")
    if any(output == path or path.is_relative_to(output) for path in paths):
        raise ValueError("tokenizer output cannot contain its input corpus")

    def file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()

    corpus = [{"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size} for path in paths]
    # Exclude machine-local absolute paths from the immutable corpus identity.
    corpus_identity = [{"sha256": entry["sha256"], "bytes": entry["bytes"]} for entry in corpus]
    corpus_sha = hashlib.sha256(json.dumps(corpus_identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    text_digest = hashlib.sha256()
    counts = {"text_chunks": 0, "utf8_bytes": 0, "sft_records": 0, "fixture_records": 0}

    def chunks():
        for path in paths:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                for number, line in enumerate(stream, 1):
                    if input_format == "text":
                        values = [line]
                    else:
                        if not line.strip():
                            continue
                        try:
                            raw = json.loads(line)
                            if not isinstance(raw, dict):
                                raise ValueError("SFT row must be an object")
                            if raw.get("split", "train") != "train":
                                raise ValueError("tokenizer training cannot include validation or test rows")
                            record = SFTRecord.from_dict(raw)
                        except (ValueError, TypeError, KeyError) as error:
                            raise ValueError(f"{path}:{number}: {error}") from error
                        counts["sft_records"] += 1
                        counts["fixture_records"] += int(record.fixture)
                        values = [message.content + "\n" for message in record.messages]
                    for value in values:
                        encoded = value.encode("utf-8")
                        # Length framing identifies the exact iterator, not just a
                        # concatenation that could conceal different boundaries.
                        text_digest.update(len(encoded).to_bytes(8, "big"))
                        text_digest.update(encoded)
                        counts["text_chunks"] += 1
                        counts["utf8_bytes"] += len(encoded)
                        yield value

    try:
        import tokenizers
        from tokenizers import ByteLevelBPETokenizer
        from transformers import PreTrainedTokenizerFast
    except ImportError as error:
        raise ImportError("scratch tokenizer training requires the pinned tokenizers/transformers environment") from error
    backend = ByteLevelBPETokenizer(add_prefix_space=False, trim_offsets=False)
    special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<|im_start|>", "<|im_end|>"]
    backend.train_from_iterator(chunks(), vocab_size=vocab_size, min_frequency=min_frequency,
                                show_progress=False, special_tokens=special_tokens)
    if counts["utf8_bytes"] == 0:
        raise ValueError("tokenizer training corpus is empty")
    if any(file_hash(path) != entry["sha256"] for path, entry in zip(paths, corpus)):
        raise ValueError("tokenizer corpus changed during training; freeze its revision before retrying")
    actual_vocab = backend.get_vocab_size()
    if actual_vocab < vocab_size and not smoke_test:
        raise ValueError(f"corpus reaches only {actual_vocab} tokens, below requested {vocab_size}; add training data or explicitly use smoke_test=True")
    fast = PreTrainedTokenizerFast(tokenizer_object=backend._tokenizer, pad_token="<pad>",
                                  bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                                  additional_special_tokens=["<|im_start|>", "<|im_end|>"],
                                  model_max_length=2048)
    # This is the new scratch tokenizer's own template. Original Qwen templates
    # and special-token tables are never loaded or modified by this function.
    fast.chat_template = (
        "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' }}"
        "{% if message['role'] == 'assistant' %}{% generation %}"
        "{{ message['content'] + '<|im_end|>\\n' }}{% endgeneration %}"
        "{% else %}{{ message['content'] + '<|im_end|>\\n' }}{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
    output.mkdir(parents=True, exist_ok=False)
    fast.save_pretrained(str(output))
    artifacts = {str(path.name): file_hash(path) for path in sorted(output.iterdir()) if path.is_file()}
    metadata = {"schema_version": 1, "kind": "scratch_bytelevel_bpe", "smoke_test": smoke_test,
                "requested_vocab_size": vocab_size, "actual_vocab_size": actual_vocab,
                "requested_vocabulary_reached": actual_vocab == vocab_size,
                "input_format": input_format, "declared_corpus_split": "train", "corpus": corpus,
                "corpus_sha256": corpus_sha, "training_iterator_sha256": text_digest.hexdigest(),
                "counts": counts, "sources": [source.__dict__ for source in sources],
                "tokenizers_version": tokenizers.__version__, "min_frequency": min_frequency,
                "special_tokens": special_tokens, "artifact_sha256": artifacts,
                "tokenizer_json": str(output / "tokenizer.json"), "output_directory": str(output),
                "status": "smoke fixture artifact" if smoke_test else "scratch candidate tokenizer; evaluate held-out bits/byte before selection"}
    identity = {key: value for key, value in metadata.items() if key not in {"corpus", "output_directory", "tokenizer_json"}}
    metadata["revision"] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    (output / "sakura-tokenizer.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def select_experiment(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Rank equal-budget experiments by bilingual execution, then BPB, then speed.

    Required result fields: experiment_id, evaluation_revision,
    compute_budget_flops, paired_executed_problems, japanese_pass_at_1,
    english_pass_at_1, validation_bits_per_byte, throughput_tokens_per_second.
    Pass rates refer to the identical fixed paired ja/en benchmark. FLOPs budgets
    and paired counts must be equal; unlike token counts this permits a fair
    comparison across different tokenizer vocabularies.
    """
    values = [dict(result) for result in results]
    if not values:
        raise ValueError("at least one completed experiment is required")
    revisions, budgets, pair_counts, ids = set(), set(), set(), set()
    for value in values:
        identifier, revision = value.get("experiment_id"), value.get("evaluation_revision")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("experiments require distinct nonempty experiment_id values")
        if not isinstance(revision, str) or not revision:
            raise ValueError("evaluation_revision is required")
        ids.add(identifier)
        revisions.add(revision)
        budget = value.get("compute_budget_flops")
        pairs = value.get("paired_executed_problems")
        if not isinstance(budget, (int, float)) or isinstance(budget, bool) or not math.isfinite(budget) or budget <= 0:
            raise ValueError("compute_budget_flops must be positive and finite")
        if not isinstance(pairs, int) or isinstance(pairs, bool) or pairs <= 0:
            raise ValueError("ja/en execution on at least one paired problem is required")
        budgets.add(budget)
        pair_counts.add(pairs)
        for field in ("japanese_pass_at_1", "english_pass_at_1", "validation_bits_per_byte", "throughput_tokens_per_second"):
            number = value.get(field)
            if not isinstance(number, (int, float)) or isinstance(number, bool) or not math.isfinite(number) or number < 0:
                raise ValueError(f"{field} must be finite and nonnegative")
        if max(value["japanese_pass_at_1"], value["english_pass_at_1"]) > 1:
            raise ValueError("pass@1 must be in [0, 1]")
        if value["throughput_tokens_per_second"] == 0:
            raise ValueError("sustained throughput must be positive")
    if len(revisions) != 1 or len(budgets) != 1 or len(pair_counts) != 1:
        raise ValueError("selection requires the same fixed evaluation revision, paired count and compute budget")

    def score(value: Mapping[str, Any]) -> tuple[float, float, float, str]:
        bilingual = (value["japanese_pass_at_1"] + value["english_pass_at_1"]) / 2
        return (-bilingual, value["validation_bits_per_byte"], -value["throughput_tokens_per_second"], value["experiment_id"])

    winner = min(values, key=score)
    return {**winner, "balanced_pass_at_1": -score(winner)[0],
            "selection_order": ["balanced ja/en executed pass@1 (descending)", "validation bits/byte (ascending)",
                                "sustained tokens/second (descending)", "experiment_id (lexicographic)"]}


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("observation timestamps must be timezone-aware ISO-8601 strings")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("invalid observation timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("observation timestamps must include an explicit timezone")
    return parsed.astimezone(timezone.utc)


def project_schedule(
    observations: Iterable[Mapping[str, Any]], *, target_tokens: int, completed_tokens: int = 0,
    margin: float = 0.3, future_availability_fraction: float = 1.0,
) -> dict[str, Any]:
    """Project duration only from >=7 contiguous days of cluster-wide evidence.

    Each observation has started_at, ended_at, tokens, counting *committed global
    tokens*, not per-worker activations. Include zero-token downtime intervals.
    No gaps/overlaps, date guessing, start date, or claimed capacity is permitted.
    The observed wall-clock rate already contains participant availability; the
    optional future fraction only applies a further explicit availability reduction.
    """
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in (target_tokens, completed_tokens)) or completed_tokens > target_tokens:
        raise ValueError("token counts must be nonnegative integers with completed <= target")
    if not isinstance(margin, (int, float)) or isinstance(margin, bool) or not math.isfinite(margin) or not 0 <= margin < 1:
        raise ValueError("margin must be in [0, 1)")
    if not isinstance(future_availability_fraction, (int, float)) or isinstance(future_availability_fraction, bool) or not math.isfinite(future_availability_fraction) or not 0 < future_availability_fraction <= 1:
        raise ValueError("future_availability_fraction must be in (0, 1]")
    intervals = []
    for sample in observations:
        start, end = _timestamp(sample.get("started_at")), _timestamp(sample.get("ended_at"))
        tokens = sample.get("tokens")
        if end <= start or not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
            raise ValueError("observations require increasing timestamps and nonnegative integer tokens")
        intervals.append((start, end, tokens))
    intervals.sort()
    if not intervals:
        raise ValueError("at least seven days of real observations are required")
    for previous, current in zip(intervals, intervals[1:]):
        if previous[1] != current[0]:
            raise ValueError("observations must be contiguous and non-overlapping; record downtime as zero-token intervals")
    duration = (intervals[-1][1] - intervals[0][0]).total_seconds()
    if duration < timedelta(days=7).total_seconds():
        raise ValueError("at least seven full days of sustained observations are required")
    committed = sum(tokens for _, _, tokens in intervals)
    if committed <= 0:
        raise ValueError("observations contain no committed training tokens")
    observed_rate = committed / duration
    effective = observed_rate * (1 - margin) * future_availability_fraction
    remaining = target_tokens - completed_tokens
    seconds = remaining / effective
    return {"observed_days": duration / 86400, "observed_committed_tokens": committed,
            "observed_tokens_per_second": observed_rate, "effective_tokens_per_second": effective,
            "margin": margin, "future_availability_fraction": future_availability_fraction,
            "target_tokens": target_tokens, "completed_tokens": completed_tokens, "remaining_tokens": remaining,
            "projected_seconds": seconds, "projected_days": seconds / 86400,
            "basis": "measured cluster-wide committed wall-clock throughput, including downtime; capacity is a separate admission check"}
