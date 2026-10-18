"""Auditable, leakage-resistant SFT data preparation.

No tokenizer is created, resized, or modified here. Callers must load the exact
tokenizer revision from their run manifest. Chat formatting follows the original
tokenizer's template (https://huggingface.co/docs/transformers/chat_templating).
The tiny bundled records are hand-authored fixtures, never a training corpus.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import unicodedata
from typing import Any, Iterable, Mapping

IGNORE_INDEX = -100


@dataclass(frozen=True)
class SourceProvenance:
    name: str
    url: str
    revision: str
    license: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceProvenance":
        if not isinstance(value, Mapping):
            raise ValueError("source must be an object with name, url, revision and license")
        fields = {}
        for key in ("name", "url", "revision", "license"):
            item = value.get(key)
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"source.{key} must be a nonempty string")
            fields[key] = item.strip()
        return cls(**fields)


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class SFTRecord:
    id: str
    language: str
    repository: str
    problem_id: str
    messages: tuple[Message, ...]
    source: SourceProvenance
    fixture: bool = False
    additional_sources: tuple[SourceProvenance, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SFTRecord":
        required = {}
        for key in ("id", "language", "repository", "problem_id"):
            item = value.get(key)
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"{key} must be a nonempty string")
            required[key] = item.strip()
        if required["language"] not in {"ja", "en"}:
            raise ValueError("language must be ja or en")
        raw_messages = value.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError("messages must be a nonempty list")
        messages = []
        for raw in raw_messages:
            if not isinstance(raw, Mapping) or raw.get("role") not in {"system", "user", "assistant"}:
                raise ValueError("messages require system, user or assistant roles")
            content = raw.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("message content must be a nonempty string")
            if raw["role"] == "system" and messages:
                raise ValueError("system message is allowed only at the beginning")
            if raw["role"] != "system":
                expected = "assistant" if messages and messages[-1].role == "user" else "user"
                if raw["role"] != expected:
                    raise ValueError("messages must alternate user and assistant after optional system")
            messages.append(Message(raw["role"], content))
        if messages[-1].role != "assistant":
            raise ValueError("SFT records must end with an assistant response")
        fixture = value.get("fixture", False)
        if not isinstance(fixture, bool):
            raise ValueError("fixture must be boolean")
        return cls(
            **required,
            messages=tuple(messages),
            source=SourceProvenance.from_dict(value.get("source")),
            fixture=fixture,
            additional_sources=tuple(SourceProvenance.from_dict(s) for s in value.get("additional_sources", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["messages"] = list(value["messages"])
        value["additional_sources"] = list(value["additional_sources"])
        return value


def load_records(path: str | Path) -> list[SFTRecord]:
    """Read JSONL with explicit provenance; report the offending source line."""
    records = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record must be an object")
                records.append(SFTRecord.from_dict(value))
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("record ids must be unique")
    return records


def content_digest(record: SFTRecord) -> str:
    """Conservative exact-content dedup: NFC + newline normalization, preserving code indentation."""
    payload = [(message.role, unicodedata.normalize("NFC", message.content.replace("\r\n", "\n").replace("\r", "\n")))
               for message in record.messages]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def prepare_dataset(
    records: Iterable[SFTRecord],
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: str = "sakura-v1",
) -> dict[str, list[SFTRecord]]:
    """Deduplicate and assign complete repository/problem connected components.

    A repository, a globally namespaced problem_id, or duplicate content connects
    records. Connections are computed *before* deduplication, so a removed duplicate
    cannot leak a repository into another split. Adding a connected record can
    reassign that component; freeze the resulting manifest as a data revision.
    Fractions are expectations over hashed groups, not exact per-record quotas.
    """
    if not all(math.isfinite(n) and 0 <= n <= 1 for n in (train_fraction, validation_fraction)):
        raise ValueError("split fractions must be finite values in [0, 1]")
    if train_fraction + validation_fraction > 1:
        raise ValueError("train_fraction + validation_fraction must be at most 1")
    ordered = sorted(records, key=lambda record: record.id)
    if len({r.id for r in ordered}) != len(ordered):
        raise ValueError("record ids must be unique")
    parents = list(range(len(ordered)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners: dict[str, int] = {}
    digests = [content_digest(record) for record in ordered]
    keys_by_record = []
    for index, record in enumerate(ordered):
        keys = ("repository:" + record.repository.strip().rstrip("/").casefold(),
                "problem:" + record.problem_id,
                "content:" + digests[index])
        keys_by_record.append(keys)
        for key in keys:
            if key in owners:
                parents[root(index)] = root(owners[key])
            else:
                owners[key] = index
    components: dict[int, set[str]] = {}
    for index, keys in enumerate(keys_by_record):
        components.setdefault(root(index), set()).update(keys)
    assignment = {}
    for group, keys in components.items():
        digest = hashlib.sha256((seed + "\0" + "\0".join(sorted(keys))).encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        assignment[group] = "train" if value < train_fraction else "validation" if value < train_fraction + validation_fraction else "test"
    result: dict[str, list[SFTRecord]] = {"train": [], "validation": [], "test": []}
    duplicate_groups: dict[str, list[int]] = {}
    for index, digest in enumerate(digests):
        duplicate_groups.setdefault(digest, []).append(index)
    for indices in duplicate_groups.values():
        record = ordered[indices[0]]
        provenance = {s for index in indices for s in (ordered[index].source, *ordered[index].additional_sources)}
        provenance.discard(record.source)
        combined = replace(record, additional_sources=tuple(sorted(provenance, key=lambda source: (source.url, source.revision, source.license, source.name))))
        result[assignment[root(indices[0])]].append(combined)
    return result


def dataset_manifest(records: Iterable[SFTRecord], splits: Mapping[str, list[SFTRecord]]) -> dict[str, Any]:
    """Audit summary: immutable hashes, provenance, duplicate removal and held-out overlap."""
    original = list(records)
    all_retained = [r for values in splits.values() for r in values]
    split_digests = {key: {content_digest(r) for r in values} for key, values in splits.items()}
    overlaps = {f"{left}/{right}": len(split_digests[left] & split_digests[right])
                for left in splits for right in splits if left < right}
    if any(overlaps.values()):
        raise ValueError("content contamination found between splits")
    # Evaluate repository/problem overlap as well, including externally supplied splits.
    seen_groups: dict[str, str] = {}
    for split, values in splits.items():
        for record in values:
            for group in ("repo:" + record.repository.strip().rstrip("/").casefold(), "problem:" + record.problem_id):
                if group in seen_groups and seen_groups[group] != split:
                    raise ValueError(f"group contamination found: {group}")
                seen_groups[group] = split
    source_set = {s for record in original for s in (record.source, *record.additional_sources)}
    payload = {key: [r.to_dict() for r in sorted(values, key=lambda r: r.id)] for key, values in sorted(splits.items())}
    # Retained rows alone omit removed duplicate IDs and grouping metadata. Hash
    # the complete frozen input too, so those changes cannot masquerade as the
    # same data revision merely because the component stays in the same split.
    identity = {"input": [r.to_dict() for r in sorted(original, key=lambda r: r.id)], "splits": payload}
    revision = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"data_revision": revision, "input_records": len(original), "retained_records": len(all_retained),
            "duplicates_removed": len(original) - len(all_retained),
            "counts": {key: len(values) for key, values in splits.items()},
            "contains_fixtures": any(r.fixture for r in original),
            "content_overlap": overlaps,
            "sources": [asdict(s) for s in sorted(source_set, key=lambda s: (s.url, s.revision, s.license, s.name))]}


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or (value and isinstance(value[0], (list, tuple))):
        raise ValueError("tokenizer must return one unbatched token sequence")
    return [int(token) for token in value]


def _encode_assistant_labels(record: SFTRecord, tokenizer: Any) -> tuple[list[int], list[int]]:
    messages = [asdict(message) for message in record.messages]
    common = {"tokenize": True, "add_generation_prompt": False}
    # Native generation tags are the most reliable mask if the original template
    # provides them. Some Qwen templates have no generation tags; use strict token
    # prefix checks below without rewriting their original chat template.
    encoded = tokenizer.apply_chat_template(messages, return_dict=True, return_assistant_tokens_mask=True, **common)
    ids = _token_ids(encoded)
    raw_mask = encoded.get("assistant_masks") if isinstance(encoded, Mapping) else None
    if raw_mask is not None and hasattr(raw_mask, "tolist"):
        raw_mask = raw_mask.tolist()
    if raw_mask is not None and len(raw_mask) == len(ids) and any(raw_mask):
        labels = [token if is_assistant else IGNORE_INDEX for token, is_assistant in zip(ids, raw_mask)]
        return ids, labels
    labels = [IGNORE_INDEX] * len(ids)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = _token_ids(tokenizer.apply_chat_template(messages[:index], tokenize=True, add_generation_prompt=True))
        through = _token_ids(tokenizer.apply_chat_template(messages[:index + 1], **common))
        if not prefix or ids[:len(prefix)] != prefix or ids[:len(through)] != through or len(through) <= len(prefix):
            raise ValueError("original chat template has no assistant mask and is not prefix-stable; provide a verified template with generation tags; no tokenizer was modified")
        labels[len(prefix):len(through)] = ids[len(prefix):len(through)]
    return ids, labels


def encode_windows(record: SFTRecord, tokenizer: Any, max_sequence_length: int) -> list[dict[str, list[int]]]:
    """Pad independent documents; long documents use one-token context overlap.

    No packing, cache, truncation, added special tokens, or tokenizer changes.
    Labels are unshifted as expected by Hugging Face causal language models; padding,
    prompts, and each window's context token use IGNORE_INDEX.
    """
    if not isinstance(max_sequence_length, int) or isinstance(max_sequence_length, bool) or max_sequence_length < 2:
        raise ValueError("max_sequence_length must be an explicit integer >= 2")
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(pad_id, int):
        raise ValueError("original tokenizer needs an existing pad_token_id or eos_token_id")
    ids, labels = _encode_assistant_labels(record, tokenizer)
    windows = []
    for start in range(0, max(len(ids) - 1, 0), max_sequence_length - 1):
        end = min(start + max_sequence_length, len(ids))
        window_ids = ids[start:end]
        window_labels = labels[start:end]
        window_labels[0] = IGNORE_INDEX
        if all(label == IGNORE_INDEX for label in window_labels):
            continue
        padding = max_sequence_length - len(window_ids)
        windows.append({"input_ids": window_ids + [pad_id] * padding,
                        "attention_mask": [1] * len(window_ids) + [0] * padding,
                        "labels": window_labels + [IGNORE_INDEX] * padding})
    if not windows:
        raise ValueError(f"record {record.id} has no trainable assistant tokens")
    return windows


def encode_record(record: SFTRecord, tokenizer: Any, max_sequence_length: int) -> dict[str, list[int]]:
    """Encode one short record; refuse silent truncation of long records."""
    windows = encode_windows(record, tokenizer, max_sequence_length)
    if len(windows) != 1:
        raise ValueError("record requires multiple windows; use encode_windows to retain every assistant token")
    return windows[0]
