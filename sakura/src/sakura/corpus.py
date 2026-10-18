"""Raw continuous-pretraining documents and deterministic training window streams.

Source records are currently indexed in memory. Tokenization is lazy by document,
not by entire corpus, and encoded windows are not cached across documents/epochs.
This is not a claim of terabyte-scale source streaming or token indexing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from .data import (IGNORE_INDEX, Message, SFTRecord, SourceProvenance,
                   _encode_assistant_labels, _token_ids)


def load_cpt_records(path: str | Path) -> list[SFTRecord]:
    """Load provenance-bearing raw-text JSONL for continuous pretraining.

    Each row requires id, language (ja/en), repository, globally namespaced
    problem_id, text and source{name,url,revision,license}; fixture is optional.
    The sole assistant Message is an *audit representation*, not a conversation.
    It lets prepare_dataset/dataset_manifest reuse exact deduplication and grouped
    train/validation/test splitting without inserting chat-template text.
    Never pass these audit-only records to the conversational SFT loader/encoder.
    """
    records = []
    ids = set()
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("CPT row must be an object")
                fields = {}
                for key in ("id", "language", "repository", "problem_id"):
                    value = raw.get(key)
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(f"{key} must be a nonempty string")
                    fields[key] = value.strip()
                if fields["language"] not in {"ja", "en"}:
                    raise ValueError("language must be ja or en")
                text = raw.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("text must be nonempty raw document text")
                fixture = raw.get("fixture", False)
                if not isinstance(fixture, bool):
                    raise ValueError("fixture must be boolean")
                source = SourceProvenance.from_dict(raw.get("source"))
                if fields["id"] in ids:
                    raise ValueError("CPT record ids must be unique")
                ids.add(fields["id"])
                records.append(SFTRecord(**fields, messages=(Message("assistant", text),),
                                         source=source, fixture=fixture))
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
    return records


def _validate_sequence_length(sequence_length: int) -> None:
    if type(sequence_length) is not int or sequence_length < 2:
        raise ValueError("sequence length must be an explicit integer >= 2")


def _padding_id(tokenizer: Any) -> int:
    value = getattr(tokenizer, "pad_token_id", None)
    if value is None:
        value = getattr(tokenizer, "eos_token_id", None)
    if type(value) is not int or value < 0:
        raise ValueError("original tokenizer requires an existing pad_token_id or eos_token_id")
    return value


def _iter_windows(
    ids: list[int], labels: list[int], sequence_length: int, pad_id: int,
) -> Iterator[dict[str, list[int]]]:
    # Each successive document window retains exactly one context token. Its
    # label is ignored because it was already trained in the preceding window.
    # With causal LM label shifting this preserves every remaining target once.
    for start in range(0, max(len(ids) - 1, 0), sequence_length - 1):
        end = min(start + sequence_length, len(ids))
        window_ids, window_labels = ids[start:end], labels[start:end]
        window_labels[0] = IGNORE_INDEX
        if all(label == IGNORE_INDEX for label in window_labels):
            continue
        padding = sequence_length - len(window_ids)
        yield {"input_ids": window_ids + [pad_id] * padding,
               "attention_mask": [1] * len(window_ids) + [0] * padding,
               "labels": window_labels + [IGNORE_INDEX] * padding}


def encode_cpt_windows(
    record: SFTRecord, tokenizer: Any, max_sequence_length: int,
) -> Iterator[dict[str, list[int]]]:
    """Yield padded raw-text windows, with no chat template or document packing.

    Uses encode(text, add_special_tokens=False), then appends the tokenizer's
    existing EOS. No tokenizer attributes or vocabulary are changed. One document
    is tokenized at a time; its token IDs remain in memory while its windows yield.
    Labels supervise all next-token positions, including EOS, except window
    context and padding. Empty tokenized documents fail instead of training pads.
    """
    _validate_sequence_length(max_sequence_length)
    if len(record.messages) != 1 or record.messages[0].role != "assistant":
        raise ValueError("CPT requires audit-only raw document records from load_cpt_records")
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if type(eos_id) is not int or eos_id < 0:
        raise ValueError("CPT requires the tokenizer's existing integer eos_token_id")
    pad_id = _padding_id(tokenizer)
    encoded = tokenizer.encode(record.messages[0].content, add_special_tokens=False)
    # Hugging Face transformers returns a list; the low-level tokenizers library
    # returns Encoding.ids. Both remain local and retain the original token IDs.
    ids = _token_ids(encoded.ids if hasattr(encoded, "ids") else encoded) + [eos_id]
    yielded = False
    for window in _iter_windows(ids, ids, max_sequence_length, pad_id):
        yielded = True
        yield window
    if not yielded:
        raise ValueError(f"CPT record {record.id} has no trainable next-token windows")


def _sft_windows(record: SFTRecord, tokenizer: Any, sequence_length: int) -> Iterator[dict[str, list[int]]]:
    if len(record.messages) == 1 and record.messages[0].role == "assistant":
        raise ValueError("raw CPT audit records cannot be encoded with objective=sft")
    ids, labels = _encode_assistant_labels(record, tokenizer)
    yielded = False
    for window in _iter_windows(ids, labels, sequence_length, _padding_id(tokenizer)):
        yielded = True
        yield window
    if not yielded:
        raise ValueError(f"SFT record {record.id} has no trainable assistant windows")


def iter_training_windows(
    records: Iterable[SFTRecord], tokenizer: Any, seq: int, objective: str = "sft", start_cursor: int = 0,
) -> Iterator[dict[str, list[int]]]:
    """Yield an endless deterministic epoch stream resuming a global window cursor.

    Record order is the frozen dataset order, with no shuffling/packing. At cursor
    zero the first document is tokenized immediately; later documents stay raw
    until needed. A large resume cursor counts at most one full epoch before
    reducing by the epoch window count, avoiding replay of all prior epochs.
    Callers should retain the frozen manifest revision and checkpoint the count
    of successfully committed windows, never the number merely prefetched.
    """
    _validate_sequence_length(seq)
    if objective not in {"sft", "cpt"}:
        raise ValueError("objective must be sft or cpt")
    if type(start_cursor) is not int or start_cursor < 0:
        raise ValueError("start_cursor must be a nonnegative integer global window count")
    # Source indexing is deliberate and explicit; encoded windows are never
    # accumulated. A one-pass input iterable can thus repeat without exhaustion.
    ordered = tuple(records)
    if not ordered:
        raise ValueError("training corpus has no records or trainable windows")
    remaining = start_cursor
    encoder = encode_cpt_windows if objective == "cpt" else _sft_windows
    while True:
        epoch_windows = 0
        for record in ordered:
            for window in encoder(record, tokenizer, seq):
                epoch_windows += 1
                if remaining:
                    remaining -= 1
                    continue
                yield window
        if epoch_windows == 0:
            raise ValueError("training corpus has no trainable windows")
        if remaining:
            remaining %= epoch_windows
