"""Content identities for local model/tokenizer snapshots.

An upstream revision label is provenance, not proof of local file contents.
Fingerprints contain portable relative names, byte lengths and streaming SHA256;
they contain no absolute machine paths. Symlinks and Windows reparse points are
rejected instead of silently hashing a different directory or cache blob.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Mapping


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BLOCK_SIZE = 4 * 1024 * 1024
_TOKENIZER_NAMES = frozenset({
    "added_tokens.json", "special_tokens_map.json", "spiece.model",
    "sentencepiece.model", "sentencepiece.bpe.model", "tekken.json",
    "tiktoken.model", "chat_template.jinja", "chat_template.json",
})


def _is_link(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _root(path) -> Path:
    root = Path(path).absolute()
    if not root.is_dir():
        raise ValueError("Snapshot must be an existing local directory")
    for component in (root, *root.parents):
        if _is_link(component):
            raise ValueError("Snapshot paths must not contain symlinks or reparse points")
    return root


def _relative_name(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\0" in name:
        raise ValueError("Artifact names must be canonical relative POSIX filenames")
    if name.startswith("/") or any(part in {"", ".", ".."} for part in name.split("/")):
        raise ValueError("Artifact path traversal or non-canonical name")
    if PurePosixPath(name).is_absolute():
        raise ValueError("Artifact name must be relative")
    return name


def _walk(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    folded: set[str] = set()
    for directory, directories, filenames in os.walk(root, followlinks=False):
        parent = Path(directory)
        # Reject even unselected links: no caller should accidentally traverse a
        # newly added cache link while changing artifact selection in the future.
        for name in directories + filenames:
            entry = parent / name
            if _is_link(entry):
                raise ValueError("Snapshot contains a symlink or reparse point")
        for name in filenames:
            file = parent / name
            if not file.is_file():
                raise ValueError("Snapshot contains a non-regular file")
            relative = _relative_name(file.relative_to(root).as_posix())
            if relative.casefold() in folded:
                raise ValueError("Case-colliding artifact names are not portable")
            folded.add(relative.casefold())
            files[relative] = file
    return files


def _tokenizer_name(name: str) -> bool:
    path = PurePosixPath(name)
    base = path.name.casefold()
    return (base == "config.json" or base.startswith("tokenizer")
            or base.startswith(("vocab.", "merges.")) or base in _TOKENIZER_NAMES
            or base.startswith("chat_template.")
            or ("chat_templates" in [part.casefold() for part in path.parts[:-1]]
                and path.suffix.casefold() in {".jinja", ".json"}))


def _snapshot_name(name: str) -> bool:
    base = PurePosixPath(name).name.casefold()
    return (_tokenizer_name(name) or base.endswith(".safetensors")
            or base.endswith(".safetensors.index.json")
            or base in {"generation_config.json", "sakura-tokenizer.json"})


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("Duplicate JSON object key in artifact metadata")
        result[name] = value
    return result


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON artifact metadata: {path.name}") from error


def _validate_indexes(files: Mapping[str, Path]) -> None:
    for name, file in files.items():
        if not name.casefold().endswith(".safetensors.index.json"):
            continue
        raw = _read_json(file)
        weights = raw.get("weight_map") if isinstance(raw, dict) else None
        if not isinstance(weights, dict) or not weights:
            raise ValueError("Safetensors index must contain a nonempty weight_map")
        for tensor, shard in weights.items():
            if not isinstance(tensor, str) or not tensor:
                raise ValueError("Invalid safetensors tensor name")
            shard = _relative_name(shard)
            if not shard.casefold().endswith(".safetensors") or shard not in files:
                raise ValueError("Safetensors index references a missing or invalid local shard")


def _signature(path: Path):
    info = path.stat()
    return info.st_size, info.st_mtime_ns, info.st_ino


def _hash_file(path: Path) -> dict:
    if _is_link(path):
        raise ValueError("Artifact became a symlink or reparse point")
    before = _signature(path)
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while block := stream.read(_BLOCK_SIZE):
            digest.update(block)
            size += len(block)
    if _is_link(path) or before != _signature(path) or size != before[0]:
        raise ValueError("Artifact changed while it was being fingerprinted")
    return {"sha256": digest.hexdigest(), "bytes": size}


def _collect(root: Path, tokenizer: bool) -> dict[str, Path]:
    files = _walk(root)
    choose = _tokenizer_name if tokenizer else _snapshot_name
    selected = {name: file for name, file in files.items() if choose(name)}
    if tokenizer:
        if not any(name != "config.json" for name in selected):
            raise ValueError("Snapshot has no tokenizer artifacts")
    else:
        if "config.json" not in selected:
            raise ValueError("Model snapshot is missing config.json")
        if not any(name.casefold().endswith(".safetensors") for name in selected):
            raise ValueError("Model snapshot has no safetensors weights")
        _validate_indexes(files)
    return dict(sorted(selected.items()))


def _fingerprint(path, tokenizer: bool) -> dict[str, dict]:
    root = _root(path)
    files = _collect(root, tokenizer)
    signatures = {name: _signature(file) for name, file in files.items()}
    result = {name: _hash_file(file) for name, file in files.items()}
    after = _collect(root, tokenizer)
    if files.keys() != after.keys() or any(signatures[name] != _signature(after[name]) for name in files):
        raise ValueError("Snapshot changed while it was being fingerprinted")
    return result


def fingerprint_snapshot(path) -> dict[str, dict]:
    """Pin model weights/index, config and all recognized tokenizer artifacts."""
    return _fingerprint(path, tokenizer=False)


def fingerprint_tokenizer(path) -> dict[str, dict]:
    """Pin semantic tokenizer files (including chat templates), not model weights."""
    return _fingerprint(path, tokenizer=True)


def _expected_map(fingerprints: Mapping, *, hashes_only: bool = False) -> dict:
    if not isinstance(fingerprints, Mapping) or not fingerprints:
        raise ValueError("Expected a nonempty pinned artifact map")
    result, folded = {}, set()
    for name, entry in fingerprints.items():
        name = _relative_name(name)
        if name.casefold() in folded:
            raise ValueError("Case-colliding expected artifact names")
        folded.add(name.casefold())
        if hashes_only and isinstance(entry, str):
            digest = entry
            length = None
        elif isinstance(entry, Mapping):
            digest, length = entry.get("sha256"), entry.get("bytes")
            if type(length) is not int or length < 0:
                raise ValueError("Artifact byte length must be a nonnegative integer")
        else:
            raise ValueError("Invalid artifact fingerprint entry")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("Artifact SHA256 must be 64 lowercase hexadecimal characters")
        result[name] = {"sha256": digest, "bytes": length}
    return dict(sorted(result.items()))


def _compare(actual: Mapping, expected: Mapping) -> None:
    missing, extra = set(expected) - set(actual), set(actual) - set(expected)
    if missing or extra:
        raise ValueError(f"Artifact list mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    for name in expected:
        wanted, found = expected[name], actual[name]
        if wanted["sha256"] != found["sha256"] or (
                wanted["bytes"] is not None and wanted["bytes"] != found["bytes"]):
            raise ValueError(f"Artifact content mismatch: {name}")


def verify_snapshot(path, fingerprints: Mapping) -> None:
    """Compare the exact selected artifact set and streamed file contents."""
    expected = _expected_map(fingerprints)
    _compare(fingerprint_snapshot(path), expected)


def verify_tokenizer(spec: Mapping, path) -> None:
    """Verify an original pinned tokenizer or ``train_tokenizer`` scratch output.

    Original manifests supply ``artifacts`` (SHA256/bytes map). Scratch manifests
    supply their trusted ``revision`` and may also supply ``artifact_sha256``.
    The scratch metadata revision is recomputed exactly as in train_tokenizer;
    machine-local source/output paths are excluded from its identity.
    """
    if not isinstance(spec, Mapping):
        raise ValueError("Tokenizer specification must be a mapping")
    if spec.get("kind") == "byte":
        if spec.get("revision") != "fixture-utf8-v1":
            raise ValueError("Unknown byte fixture tokenizer revision")
        return
    root = _root(path)
    metadata_path = root / "sakura-tokenizer.json"
    is_scratch = spec.get("kind") == "scratch_bytelevel_bpe" or metadata_path.is_file()
    if is_scratch:
        if not metadata_path.is_file() or _is_link(metadata_path):
            raise ValueError("Scratch tokenizer metadata must be a regular local file")
        metadata = _read_json(metadata_path)
        if metadata.get("kind") != "scratch_bytelevel_bpe" or metadata.get("schema_version") != 1:
            raise ValueError("Unsupported scratch tokenizer metadata")
        identity = {key: value for key, value in metadata.items()
                    if key not in {"revision", "corpus", "output_directory", "tokenizer_json"}}
        revision = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                             ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
        if metadata.get("revision") != revision or spec.get("revision") != revision:
            raise ValueError("Scratch tokenizer revision does not match its metadata and manifest")
        expected = _expected_map(metadata.get("artifact_sha256"), hashes_only=True)
        pinned = spec.get("artifacts", spec.get("files", spec.get("artifact_sha256")))
        if pinned is not None:
            declared = _expected_map(pinned, hashes_only=True)
            if {name: item["sha256"] for name, item in declared.items()} != {
                    name: item["sha256"] for name, item in expected.items()}:
                raise ValueError("Scratch tokenizer artifact map differs from the manifest")
            expected = declared
    else:
        expected = _expected_map(spec.get("artifacts", spec.get("files", spec.get("artifact_sha256"))), hashes_only=True)
    _compare(fingerprint_tokenizer(root), expected)
