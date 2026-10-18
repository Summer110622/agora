"""Atomic all-stage checkpoints: safetensors data and explicit JSON metadata."""
from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import threading
from pathlib import Path
from typing import Any, Mapping

import torch

from .auth import canonical_json, strict_json
from .transport import decode_value, encode_value


class CheckpointError(ValueError):
    pass


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _sync_directory(path: Path) -> None:
    # Windows does not offer POSIX directory fsync. os.replace still gives an
    # atomic pointer replacement; no power-loss durability claim is made there.
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class CheckpointManager:
    """Commit quiesced worker states as one complete, verifiable generation.

    The caller must pause dispatch, finish outstanding backward calls, and collect
    all stages at the same step before calling commit. Only CURRENT is loadable;
    orphan generations and temporary directories are never implicitly resumed.
    """
    def __init__(self, directory: str | Path, mirrors: tuple[str | Path, ...] | list[str | Path] = (),
                 *, max_shard_bytes: int = 64 * 1024 * 1024 * 1024):
        self.directory = Path(directory).resolve()
        self.mirrors = tuple(Path(path).resolve() for path in mirrors)
        roots = [self.directory, *self.mirrors]
        for index, root in enumerate(roots):
            for other in roots[:index]:
                if root == other or root.is_relative_to(other) or other.is_relative_to(root):
                    raise ValueError("Checkpoint roots must be distinct, non-nested directories")
        if max_shard_bytes < 1024:
            raise ValueError("Checkpoint shard limit is too small")
        self.max_shard_bytes = max_shard_bytes
        self._lock = threading.Lock()
        self.last_mirror_errors: list[str] = []

    def mirror_diagnostics(self) -> dict[str, Any]:
        return {"primary": str(self.directory), "mirrors": [str(path) for path in self.mirrors],
                "distinct_directories": True, "physical_host_redundancy_verified": False,
                "errors": list(self.last_mirror_errors)}

    def commit(self, stage_states: Mapping[str, dict[str, Any]], training_state: dict[str, Any],
               *, run_hash: str, step: int) -> Path:
        if type(step) is not int or step < 0 or not stage_states:
            raise CheckpointError("Checkpoint requires a nonnegative step and every stage")
        if not isinstance(run_hash, str) or len(run_hash) != 64:
            raise CheckpointError("Checkpoint requires a manifest SHA-256 digest")
        try:
            bytes.fromhex(run_hash)
        except ValueError as error:
            raise CheckpointError("Invalid run hash") from error
        if training_state.get("step") != step:
            raise CheckpointError("Trainer snapshot step does not match checkpoint step")
        for stage, state in stage_states.items():
            if not isinstance(stage, str) or not stage or not isinstance(state, dict) or state.get("step") != step:
                raise CheckpointError("All stage snapshots must declare the same checkpoint step")
            if state.get("run_hash") != run_hash:
                raise CheckpointError("Stage snapshot belongs to a different run manifest")
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            generation = f"step-{step:012d}-{secrets.token_hex(8)}"
            temporary = self.directory / f".{generation}.tmp"
            final = self.directory / generation
            temporary.mkdir()
            try:
                entries: dict[str, dict[str, Any]] = {}
                for index, (stage, state) in enumerate(sorted(stage_states.items())):
                    entries[stage] = self._save_state(temporary, f"stage-{index:06d}", state)
                training_entry = self._save_state(temporary, "training", training_state)
                manifest = {"format": 2, "generation": generation, "run_hash": run_hash, "step": step,
                            "stages": entries, "training_state": training_entry}
                manifest_data = canonical_json(manifest)
                _write(temporary / "manifest.json", manifest_data)
                # The commit marker and pointer bind the exact manifest bytes.
                manifest_digest = hashlib.sha256(manifest_data).hexdigest()
                _write(temporary / "COMMITTED", manifest_digest.encode("ascii"))
                _sync_directory(temporary)
                os.replace(temporary, final)
                _sync_directory(self.directory)
                pointer = canonical_json({"generation": generation, "manifest_sha256": manifest_digest})
                # Mirrors are prepared before the primary pointer is published.
                # A mirror failure aborts commit, leaving primary CURRENT intact.
                self.last_mirror_errors = []
                for mirror in self.mirrors:
                    try:
                        self._replicate(final, mirror)
                    except Exception as error:
                        self.last_mirror_errors.append(f"{mirror}: {type(error).__name__}: {error}")
                        raise CheckpointError("Mirror replication failed; primary CURRENT was not advanced") from error
                # Publish the primary only after every complete shard set exists.
                self._publish_pointer(self.directory, pointer)
                for mirror in self.mirrors:
                    try:
                        self._publish_pointer(mirror, pointer)
                    except Exception as error:
                        # The generation is durably committed on primary. Mirror
                        # pointer repair can be retried without changing its data.
                        self.last_mirror_errors.append(f"{mirror}: pointer update failed: {error}")
                return final / "manifest.json"
            except Exception:
                if temporary.exists():
                    shutil.rmtree(temporary)
                raise

    def _replicate(self, source: Path, mirror: Path) -> None:
        mirror.mkdir(parents=True, exist_ok=True)
        temporary = mirror / f".{source.name}.tmp"
        final = mirror / source.name
        if temporary.exists() or final.exists():
            raise CheckpointError("Mirror generation path already exists")
        try:
            shutil.copytree(source, temporary)
            for path in temporary.iterdir():
                if not path.is_file() or _digest(path) != _digest(source / path.name):
                    raise CheckpointError("Mirror verification failed")
                with path.open("r+b") as stream:
                    os.fsync(stream.fileno())
            _sync_directory(temporary)
            os.replace(temporary, final)
            _sync_directory(mirror)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    def _publish_pointer(self, directory: Path, data: bytes) -> None:
        path = directory / f".CURRENT-{secrets.token_hex(8)}.tmp"
        try:
            _write(path, data)
            os.replace(path, directory / "CURRENT")
            _sync_directory(directory)
        finally:
            path.unlink(missing_ok=True)

    def load_current(self, expected_run_hash: str | None = None) -> dict[str, Any]:
        pointer_path = self.directory / "CURRENT"
        if not pointer_path.exists():
            raise CheckpointError("No committed CURRENT checkpoint exists")
        if pointer_path.stat().st_size > 4096:
            raise CheckpointError("Invalid CURRENT checkpoint pointer")
        try:
            pointer = strict_json(pointer_path.read_bytes())
            if not isinstance(pointer, dict) or set(pointer) != {"generation", "manifest_sha256"}:
                raise CheckpointError("Invalid CURRENT checkpoint pointer")
            name = pointer["generation"]
            if not isinstance(name, str) or Path(name).name != name or not name.startswith("step-") or "/" in name or "\\" in name:
                raise CheckpointError("Invalid checkpoint generation path")
            generation = self.directory / name
            if generation.is_symlink() or not generation.is_dir():
                raise CheckpointError("Invalid checkpoint generation directory")
            manifest_path = generation / "manifest.json"
            if manifest_path.is_symlink() or manifest_path.stat().st_size > 4 * 1024 * 1024:
                raise CheckpointError("Invalid checkpoint manifest")
            manifest_data = manifest_path.read_bytes()
            digest = hashlib.sha256(manifest_data).hexdigest()
            marker = generation / "COMMITTED"
            if marker.is_symlink() or marker.stat().st_size != 64:
                raise CheckpointError("Invalid checkpoint commit marker")
            if (digest != pointer["manifest_sha256"]
                    or marker.read_text("ascii") != digest):
                raise CheckpointError("Checkpoint manifest or commit marker hash mismatch")
            manifest = strict_json(manifest_data)
            if (manifest.get("format") != 2 or manifest.get("generation") != name
                    or not isinstance(manifest.get("stages"), dict) or not manifest["stages"]):
                raise CheckpointError("Unsupported checkpoint manifest")
            if expected_run_hash is not None and manifest["run_hash"] != expected_run_hash:
                raise CheckpointError("Checkpoint belongs to a different run manifest")
            stages = {stage: self._load_state(generation, entry) for stage, entry in manifest["stages"].items()}
            training_state = self._load_state(generation, manifest["training_state"])
            if training_state.get("step") != manifest["step"] or any(state.get("step") != manifest["step"] for state in stages.values()):
                raise CheckpointError("Checkpoint contains inconsistent stage steps")
            if any(state.get("run_hash") != manifest["run_hash"] for state in stages.values()):
                raise CheckpointError("Checkpoint contains states from a different run manifest")
            return {"stages": stages, "training_state": training_state, "manifest": manifest}
        except CheckpointError:
            raise
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            raise CheckpointError(f"Invalid or incomplete checkpoint: {error}") from error

    def _save_state(self, directory: Path, prefix: str, state: dict[str, Any]) -> dict[str, Any]:
        from safetensors.torch import save_file
        tensors: dict[str, torch.Tensor] = {}
        total_bytes = 0
        storage_pointers: set[int] = set()

        def tensor_reference(tensor: torch.Tensor) -> dict[str, str]:
            nonlocal total_bytes
            total_bytes += tensor.numel() * tensor.element_size()
            if total_bytes > self.max_shard_bytes:
                raise CheckpointError("Checkpoint tensor shard exceeds configured limit")
            name = f"tensor-{len(tensors):08d}"
            # Keep ordinary contiguous snapshots without another full-stage
            # copy. Clone only views sharing storage with an earlier tensor.
            prepared = tensor.detach().contiguous()
            pointer = prepared.untyped_storage().data_ptr()
            if pointer and pointer in storage_pointers:
                prepared = prepared.clone()
                pointer = prepared.untyped_storage().data_ptr()
            storage_pointers.add(pointer)
            tensors[name] = prepared
            return {"kind": "tensor_ref", "name": name}

        tree = encode_value(state, max_tensor_bytes=self.max_shard_bytes, tensor_encoder=tensor_reference)
        metadata = canonical_json({"format": 1, "tree": tree})
        if len(metadata) > 16 * 1024 * 1024:
            raise CheckpointError("Checkpoint structure metadata exceeds limit")
        metadata_path = directory / f"{prefix}.json"
        _write(metadata_path, metadata)
        tensor_path = directory / f"{prefix}.safetensors"
        save_file(tensors, str(tensor_path))
        if tensor_path.stat().st_size > self.max_shard_bytes:
            raise CheckpointError("Checkpoint tensor file exceeds configured limit")
        with tensor_path.open("r+b") as stream:
            os.fsync(stream.fileno())
        return {"structure": self._describe(metadata_path), "tensors": self._describe(tensor_path)}

    @staticmethod
    def _describe(path: Path) -> dict[str, Any]:
        return {"file": path.name, "sha256": _digest(path), "bytes": path.stat().st_size}

    def _verified_path(self, generation: Path, entry: dict[str, Any]) -> Path:
        if not isinstance(entry, dict) or set(entry) != {"file", "sha256", "bytes"}:
            raise CheckpointError("Invalid shard descriptor")
        name, length = entry["file"], entry["bytes"]
        if (not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name
                or type(length) is not int or not 0 <= length <= self.max_shard_bytes):
            raise CheckpointError("Invalid shard path or length")
        path = generation / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != length or _digest(path) != entry["sha256"]:
            raise CheckpointError("Checkpoint shard is missing, truncated, or corrupted")
        return path

    def _load_state(self, generation: Path, entry: dict[str, Any]) -> dict[str, Any]:
        from safetensors.torch import load_file
        if not isinstance(entry, dict) or set(entry) != {"structure", "tensors"}:
            raise CheckpointError("Invalid state shard descriptor")
        metadata_path = self._verified_path(generation, entry["structure"])
        tensor_path = self._verified_path(generation, entry["tensors"])
        if metadata_path.stat().st_size > 16 * 1024 * 1024:
            raise CheckpointError("Checkpoint structure metadata exceeds limit")
        metadata = strict_json(metadata_path.read_bytes())
        if not isinstance(metadata, dict) or set(metadata) != {"format", "tree"} or metadata["format"] != 1:
            raise CheckpointError("Unsupported state structure format")
        tensors = load_file(str(tensor_path), device="cpu")
        used: set[str] = set()

        def tensor_from_reference(node: Any) -> torch.Tensor:
            if (not isinstance(node, dict) or set(node) != {"kind", "name"}
                    or not isinstance(node["name"], str) or node["name"] not in tensors):
                raise CheckpointError("Invalid checkpoint tensor reference")
            used.add(node["name"])
            return tensors[node["name"]]

        value = decode_value(metadata["tree"], max_tensor_bytes=self.max_shard_bytes,
                             tensor_decoder=tensor_from_reference)
        if used != set(tensors):
            raise CheckpointError("Checkpoint has unreferenced tensor data")
        if not isinstance(value, dict):
            raise CheckpointError("Checkpoint shard must contain a state dictionary")
        return value
