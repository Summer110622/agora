"""Known-member Ed25519 authentication for an independent Sakura network."""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


class AuthenticationError(ValueError):
    """A signed message was invalid, stale, replayed, or from an unknown member."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def strict_json(data: bytes | str) -> Any:
    return json.loads(data, object_pairs_hook=_unique_object,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON number")))


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Hash the complete run manifest, including membership and model configuration."""
    return hashlib.sha256(canonical_json(dict(manifest))).hexdigest()


@dataclass(frozen=True)
class Identity:
    member_id: str
    private_key: Ed25519PrivateKey

    def __post_init__(self) -> None:
        if not isinstance(self.member_id, str) or not 1 <= len(self.member_id) <= 128:
            raise ValueError("member_id must contain 1 to 128 characters")

    @classmethod
    def generate(cls, member_id: str) -> "Identity":
        return cls(member_id, Ed25519PrivateKey.generate())

    @property
    def public_key(self) -> str:
        raw = self.private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        return base64.b64encode(raw).decode("ascii")

    @property
    def public_key_b64(self) -> str:
        return self.public_key

    def save(self, path: str | Path) -> None:
        """Create a new private identity file; never overwrite an existing identity."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = self.private_key.private_bytes(serialization.Encoding.Raw,
                                            serialization.PrivateFormat.Raw,
                                            serialization.NoEncryption())
        payload = canonical_json({"format": 1, "member_id": self.member_id,
                                  "private_key": base64.b64encode(raw).decode("ascii")})
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    @classmethod
    def load(cls, path: str | Path) -> "Identity":
        path = Path(path)
        if path.stat().st_size > 4096:
            raise ValueError("Identity file is unexpectedly large")
        data = strict_json(path.read_bytes())
        if not isinstance(data, dict) or set(data) != {"format", "member_id", "private_key"} or data["format"] != 1:
            raise ValueError("Unsupported identity format")
        raw = base64.b64decode(data["private_key"], validate=True)
        return cls(data["member_id"], Ed25519PrivateKey.from_private_bytes(raw))


class MessageAuthenticator:
    """Sign and verify request/response envelopes bound to a run and request nonce.

    Replay protection is process-local. Restarting a service resets its nonce cache;
    the timestamp window still bounds admissible messages after a restart.
    """
    _fields = {"version", "kind", "member_id", "run_hash", "timestamp", "nonce",
               "in_reply_to", "payload", "signature"}

    def __init__(self, identity: Identity, allowed_members: Mapping[str, str], run_hash: str,
                 *, max_clock_skew: float = 120, replay_capacity: int = 100_000):
        if not isinstance(run_hash, str) or len(run_hash) != 64:
            raise ValueError("run_hash must be a SHA-256 hex digest")
        try:
            bytes.fromhex(run_hash)
        except ValueError as error:
            raise ValueError("run_hash must be a SHA-256 hex digest") from error
        if max_clock_skew <= 0 or replay_capacity < 1:
            raise ValueError("Invalid replay protection limits")
        self.identity, self.run_hash = identity, run_hash
        self.max_clock_skew, self.replay_capacity = max_clock_skew, replay_capacity
        self.members: dict[str, Ed25519PublicKey] = {}
        for member, public_key in allowed_members.items():
            try:
                self.members[member] = Ed25519PublicKey.from_public_bytes(
                    base64.b64decode(public_key, validate=True))
            except (ValueError, TypeError, binascii.Error) as error:
                raise ValueError(f"Invalid Ed25519 key for member {member!r}") from error
        if allowed_members.get(identity.member_id) != identity.public_key:
            raise ValueError("Local identity must appear in allowed_members with its exact public key")
        self._seen: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()

    def sign(self, payload: Any, *, kind: str = "request", in_reply_to: str | None = None,
             timestamp: float | None = None, nonce: str | None = None) -> dict[str, Any]:
        if kind not in {"request", "response"}:
            raise ValueError("Invalid message kind")
        unsigned = {"version": 1, "kind": kind, "member_id": self.identity.member_id,
                    "run_hash": self.run_hash, "timestamp": time.time() if timestamp is None else timestamp,
                    "nonce": nonce or secrets.token_hex(24), "in_reply_to": in_reply_to,
                    "payload": payload}
        signature = self.identity.private_key.sign(canonical_json(unsigned))
        return {**unsigned, "signature": base64.b64encode(signature).decode("ascii")}

    def verify(self, envelope: Any, *, kind: str = "request", expected_member: str | None = None,
               in_reply_to: str | None = None) -> Any:
        if not isinstance(envelope, dict) or set(envelope) != self._fields:
            raise AuthenticationError("Invalid signed envelope")
        member = envelope["member_id"]
        if not isinstance(member, str) or member not in self.members:
            raise AuthenticationError("Unknown member")
        if expected_member is not None and member != expected_member:
            raise AuthenticationError("Unexpected responding member")
        if (type(envelope["version"]) is not int or envelope["version"] != 1 or envelope["kind"] != kind
                or envelope["run_hash"] != self.run_hash or envelope["in_reply_to"] != in_reply_to):
            raise AuthenticationError("Message context does not match the run or request")
        timestamp, nonce = envelope["timestamp"], envelope["nonce"]
        try:
            valid_timestamp = (type(timestamp) in (float, int) and math.isfinite(timestamp)
                               and abs(time.time() - timestamp) <= self.max_clock_skew)
        except OverflowError:
            valid_timestamp = False
        if not valid_timestamp:
            raise AuthenticationError("Message timestamp is outside the permitted window")
        if not isinstance(nonce, str) or len(nonce) != 48:
            raise AuthenticationError("Invalid message nonce")
        try:
            if len(bytes.fromhex(nonce)) != 24:
                raise ValueError("Invalid nonce length")
            unsigned = {key: value for key, value in envelope.items() if key != "signature"}
            signature = base64.b64decode(envelope["signature"], validate=True)
            self.members[member].verify(signature, canonical_json(unsigned))
        except (ValueError, TypeError, InvalidSignature, binascii.Error) as error:
            raise AuthenticationError("Invalid message signature") from error
        with self._lock:
            now = time.time()
            self._seen = {key: expires for key, expires in self._seen.items() if expires >= now}
            key = (member, nonce)
            if key in self._seen:
                raise AuthenticationError("Replayed message")
            if len(self._seen) >= self.replay_capacity:
                raise AuthenticationError("Replay cache is full; retry after the timestamp window")
            self._seen[key] = float(timestamp) + self.max_clock_skew
        return envelope["payload"]
