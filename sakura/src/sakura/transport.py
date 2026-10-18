"""Portable signed HTTP RPC and bounded, explicit tensor serialization.

HTTP authenticates messages but does not encrypt activations, tokens, or gradients.
Use TLS certificate options, a trusted private network, or a configured TLS proxy.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import ipaddress
import math
import secrets
import ssl
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from .auth import AuthenticationError, Identity, MessageAuthenticator, canonical_json, strict_json

MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
MAX_BLOB_BYTES = 1024 * 1024 * 1024
_DTYPES = {name: getattr(torch, name) for name in (
    "bool", "uint8", "int8", "int16", "int32", "int64", "float16", "bfloat16", "float32", "float64")}
_NUMPY_DTYPES = {"bool", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64", "float16", "float32", "float64"}


class SerializationError(ValueError):
    pass


class RpcError(RuntimeError):
    pass


class _BlobStore:
    """Bounded expiring spool files; every transfer belongs to one signed member."""
    def __init__(self, max_blob_bytes: int, *, ttl: float = 300):
        self.max_blob_bytes, self.ttl = max_blob_bytes, ttl
        self._temporary: tempfile.TemporaryDirectory | None = None
        self._records: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._closed = False

    def _expire(self) -> None:
        for identifier, record in list(self._records.items()):
            if record["expires"] < time.monotonic():
                record["path"].unlink(missing_ok=True)
                del self._records[identifier]

    def _record(self, member: str, identifier: str) -> dict[str, Any]:
        if self._closed:
            raise RpcError("RPC blob store has closed")
        self._expire()
        if not isinstance(identifier, str) or identifier not in self._records:
            raise RpcError("Unknown or expired blob transfer")
        record = self._records[identifier]
        if record["member"] != member:
            raise AuthenticationError("Blob transfer belongs to another member")
        record["expires"] = time.monotonic() + self.ttl
        return record

    def begin(self, member: str, size: int, digest: str) -> dict[str, Any]:
        if (type(size) is not int or not 0 < size <= self.max_blob_bytes
                or not isinstance(digest, str) or len(digest) != 64):
            raise SerializationError("Invalid blob length or SHA256 digest")
        try:
            if len(bytes.fromhex(digest)) != 32:
                raise ValueError("Invalid digest")
        except ValueError as error:
            raise SerializationError("Invalid blob SHA256 digest") from error
        with self._lock:
            if self._closed:
                raise RpcError("RPC blob store has closed")
            self._expire()
            if len(self._records) >= 16 or sum(record["size"] for record in self._records.values()) + size > 2 * self.max_blob_bytes:
                raise RpcError("RPC blob spool capacity exceeded")
            if self._temporary is None:
                self._temporary = tempfile.TemporaryDirectory(prefix="sakura-rpc-")
            identifier = secrets.token_hex(24)
            path = Path(self._temporary.name) / identifier
            path.touch(exist_ok=False)
            self._records[identifier] = {"member": member, "path": path, "size": size, "sha256": digest,
                                         "expires": time.monotonic() + self.ttl}
            return {"id": identifier, "bytes": size, "sha256": digest}

    def append(self, member: str, identifier: str, offset: int, data: bytes) -> None:
        with self._lock:
            record = self._record(member, identifier)
            if (type(offset) is not int or offset != record["path"].stat().st_size
                    or not data or offset + len(data) > record["size"]):
                raise RpcError("Blob chunks must arrive in order with the declared total size")
            with record["path"].open("ab") as stream:
                stream.write(data)

    def put(self, member: str, data: bytes) -> dict[str, Any]:
        descriptor = self.begin(member, len(data), hashlib.sha256(data).hexdigest())
        self.append(member, descriptor["id"], 0, data)
        return descriptor

    def read(self, member: str, identifier: str, offset: int, length: int) -> bytes:
        with self._lock:
            record = self._record(member, identifier)
            if (type(offset) is not int or type(length) is not int or offset < 0 or not 0 < length <= 4 * 1024 * 1024
                    or offset >= record["size"] or record["path"].stat().st_size != record["size"]):
                raise RpcError("Invalid blob read range or incomplete upload")
            with record["path"].open("rb") as stream:
                stream.seek(offset)
                return stream.read(min(length, record["size"] - offset))

    def take(self, member: str, identifier: str) -> bytes:
        with self._lock:
            record = self._record(member, identifier)
            if record["path"].stat().st_size != record["size"]:
                raise RpcError("Blob upload is incomplete")
            digest = hashlib.sha256()
            with record["path"].open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != record["sha256"]:
                raise RpcError("Blob SHA256 verification failed")
            data = record["path"].read_bytes()
            record["path"].unlink()
            del self._records[identifier]
            return data

    def drop(self, member: str, identifier: str) -> None:
        with self._lock:
            record = self._record(member, identifier)
            record["path"].unlink(missing_ok=True)
            del self._records[identifier]

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._records.clear()
            if self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None


def encode_value(value: Any, *, max_tensor_bytes: int = MAX_PAYLOAD_BYTES, _depth: int = 0,
                 tensor_encoder: Callable[[torch.Tensor], Any] | None = None,
                 _budget: list[int] | None = None) -> Any:
    """Encode a tensor tree without pickle, executable objects, or ambiguous tags."""
    if _depth > 64:
        raise SerializationError("Object nesting limit exceeded")
    budget = [max_tensor_bytes] if _budget is None else _budget
    child = lambda item: encode_value(item, max_tensor_bytes=max_tensor_bytes, _depth=_depth + 1,
                                     tensor_encoder=tensor_encoder, _budget=budget)
    if value is None or type(value) in (bool, int, str):
        return {"kind": "scalar", "value": value}
    if type(value) is float:
        if not math.isfinite(value):
            raise SerializationError("Non-finite scalar")
        return {"kind": "scalar", "value": value}
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise SerializationError("RPC tensors must be dense CPU tensors")
        name = str(value.dtype).removeprefix("torch.")
        if name not in _DTYPES:
            raise SerializationError(f"Unsupported tensor dtype {name}")
        tensor_bytes = value.numel() * value.element_size()
        if tensor_bytes > min(max_tensor_bytes, budget[0]):
            raise SerializationError("Tensor byte limit exceeded")
        budget[0] -= tensor_bytes
        if tensor_encoder is not None:
            return tensor_encoder(value)
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return {"kind": "tensor", "dtype": name, "shape": list(value.shape),
                "data": base64.b64encode(raw).decode("ascii")}
    if isinstance(value, bytes):
        if len(value) > min(max_tensor_bytes, budget[0]):
            raise SerializationError("Byte value limit exceeded")
        budget[0] -= len(value)
        return {"kind": "bytes", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, (list, tuple)):
        return {"kind": "tuple" if isinstance(value, tuple) else "list", "items": [child(x) for x in value]}
    if isinstance(value, dict):
        return {"kind": "dict", "items": [[child(key), child(item)] for key, item in value.items()]}
    # NumPy is optional and only a fixed set of numeric arrays/scalars is accepted.
    if type(value).__module__.startswith("numpy"):
        import numpy as np
        if isinstance(value, np.generic):
            return child(value.item())
        if isinstance(value, np.ndarray) and value.dtype.name in _NUMPY_DTYPES:
            if value.nbytes > min(max_tensor_bytes, budget[0]):
                raise SerializationError("Array byte limit exceeded")
            budget[0] -= value.nbytes
            array = np.ascontiguousarray(value.astype(value.dtype.newbyteorder("<"), copy=False))
            return {"kind": "ndarray", "dtype": value.dtype.name, "shape": list(value.shape),
                    "data": base64.b64encode(array.tobytes()).decode("ascii")}
    raise SerializationError(f"Unsupported value type {type(value).__name__}")


def decode_value(node: Any, *, max_tensor_bytes: int = MAX_PAYLOAD_BYTES, _depth: int = 0,
                 _budget: list[int] | None = None,
                 tensor_decoder: Callable[[Any], torch.Tensor] | None = None) -> Any:
    if _depth > 64 or not isinstance(node, dict) or not isinstance(node.get("kind"), str):
        raise SerializationError("Invalid value encoding or excessive nesting")
    budget = [max_tensor_bytes] if _budget is None else _budget
    child = lambda item: decode_value(item, max_tensor_bytes=max_tensor_bytes, _depth=_depth + 1, _budget=budget,
                                     tensor_decoder=tensor_decoder)
    kind = node["kind"]
    if kind == "tensor_ref" and tensor_decoder is not None:
        return tensor_decoder(node)
    if kind == "scalar" and set(node) == {"kind", "value"}:
        value = node["value"]
        if value is None or type(value) in (bool, int, str) or (type(value) is float and math.isfinite(value)):
            return value
    elif kind in {"list", "tuple", "dict"} and set(node) == {"kind", "items"}:
        if not isinstance(node["items"], list) or len(node["items"]) > 1_000_000:
            raise SerializationError("Invalid container")
        if kind in {"list", "tuple"}:
            items = [child(item) for item in node["items"]]
            return tuple(items) if kind == "tuple" else items
        result = {}
        for pair in node["items"]:
            if not isinstance(pair, list) or len(pair) != 2:
                raise SerializationError("Invalid dictionary entry")
            key, value = child(pair[0]), child(pair[1])
            if type(key) not in (str, int, float, bool, tuple) and key is not None:
                raise SerializationError("Unsupported dictionary key")
            try:
                if key in result:
                    raise SerializationError("Duplicate dictionary key")
                result[key] = value
            except TypeError as error:
                raise SerializationError("Unhashable dictionary key") from error
        return result
    elif kind in {"tensor", "ndarray", "bytes"}:
        required = {"kind", "data"} if kind == "bytes" else {"kind", "dtype", "shape", "data"}
        if set(node) != required or not isinstance(node["data"], str):
            raise SerializationError("Invalid tensor/byte encoding")
        if len(node["data"]) > ((max_tensor_bytes + 2) // 3) * 4:
            raise SerializationError("Encoded byte limit exceeded")
        if kind != "bytes":
            shape = node["shape"]
            if not isinstance(shape, list) or len(shape) > 16 or any(type(n) is not int or n < 0 or n > max_tensor_bytes for n in shape):
                raise SerializationError("Invalid tensor shape")
            count = math.prod(shape)
            dtype = node["dtype"]
            if not isinstance(dtype, str):
                raise SerializationError("Invalid tensor dtype")
            if kind == "tensor":
                if dtype not in _DTYPES:
                    raise SerializationError("Unsupported tensor dtype")
                itemsize = torch.empty((), dtype=_DTYPES[dtype]).element_size()
            else:
                if dtype not in _NUMPY_DTYPES:
                    raise SerializationError("Unsupported array dtype")
                import numpy as np
                itemsize = np.dtype(dtype).itemsize
            if count * itemsize > min(max_tensor_bytes, budget[0]):
                raise SerializationError("Decoded tensor byte limit exceeded")
        try:
            raw = base64.b64decode(node["data"], validate=True)
        except (ValueError, binascii.Error) as error:
            raise SerializationError("Invalid base64 data") from error
        if len(raw) > budget[0]:
            raise SerializationError("Decoded aggregate byte limit exceeded")
        budget[0] -= len(raw)
        if kind == "bytes":
            return raw
        if len(raw) != count * itemsize:
            raise SerializationError("Tensor shape does not match byte length")
        if kind == "ndarray":
            return np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder("<")).copy().reshape(shape)
        if not raw:
            return torch.empty(shape, dtype=_DTYPES[dtype])
        return torch.frombuffer(bytearray(raw), dtype=_DTYPES[dtype]).clone().reshape(shape)
    raise SerializationError("Invalid value encoding")


class LocalTransport:
    def __init__(self, handler: Callable[[dict[str, Any]], dict[str, Any]]):
        self.handler = handler

    def call(self, request: dict[str, Any]) -> dict[str, Any]:
        # Copies CPU tensor trees to preserve the same ownership contract as HTTP.
        result = self.handler(decode_value(encode_value(request)))
        return decode_value(encode_value(result))


class RpcServer:
    def __init__(self, handler: Callable[[dict[str, Any]], dict[str, Any]], identity: Identity,
                 allowed_members: Mapping[str, str], run_hash: str, *, host: str = "127.0.0.1",
                 port: int = 0, allow_external: bool = False,
                 max_payload_bytes: int = MAX_PAYLOAD_BYTES, max_clock_skew: float = 120,
                 max_blob_bytes: int = MAX_BLOB_BYTES, max_concurrent_requests: int = 4,
                 pass_member: bool = False, tls_certfile: str | None = None,
                 tls_keyfile: str | None = None):
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host.lower() == "localhost"
        if not loopback and not allow_external:
            raise ValueError("External binding requires allow_external=True; HTTP is not encrypted")
        self.handler = handler
        self.pass_member = pass_member
        if bool(tls_certfile) != bool(tls_keyfile):
            raise ValueError("TLS requires both certificate and private key")
        self._tls_context = None
        if tls_certfile:
            self._tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            self._tls_context.minimum_version = ssl.TLSVersion.TLSv1_2
            self._tls_context.load_cert_chain(tls_certfile, tls_keyfile)
        self.auth = MessageAuthenticator(identity, allowed_members, run_hash, max_clock_skew=max_clock_skew)
        if max_payload_bytes < 4096 or max_blob_bytes < max_payload_bytes:
            raise ValueError("RPC wire limit must be >=4096 bytes and <= blob limit")
        if type(max_concurrent_requests) is not int or not 1 <= max_concurrent_requests <= 64:
            raise ValueError("RPC concurrency must be between 1 and 64")
        self.host, self.port, self.max_payload_bytes = host, port, max_payload_bytes
        self.max_blob_bytes = max_blob_bytes
        self.max_concurrent_requests = max_concurrent_requests
        self._blobs = _BlobStore(max_blob_bytes)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._closed = False

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("RPC server has not started")
        host, port = self._server.server_address[:2]
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        scheme = "https" if self._tls_context else "http"
        return f"{scheme}://{host}:{port}/rpc"

    def start(self) -> "RpcServer":
        if self._closed:
            raise RuntimeError("A closed RPC server cannot restart; create a new instance")
        if self._server is not None:
            return self
        owner = self
        request_slots = threading.BoundedSemaphore(self.max_concurrent_requests)

        class LimitedServer(ThreadingHTTPServer):
            def process_request(self, request, client_address):
                if not request_slots.acquire(blocking=False):
                    self.shutdown_request(request)
                    return
                try:
                    super().process_request(request, client_address)
                except BaseException:
                    request_slots.release()
                    raise

            def process_request_thread(self, request, client_address):
                try:
                    super().process_request_thread(request, client_address)
                finally:
                    request_slots.release()

        class Handler(BaseHTTPRequestHandler):
            wbufsize = 0

            def setup(self):
                super().setup()
                self.connection.settimeout(30)
                if isinstance(self.connection, ssl.SSLSocket):
                    self.connection.do_handshake()

            def log_message(self, *args):
                pass

            def do_POST(self):
                if self.path != "/rpc":
                    self.send_error(404)
                    return
                if self.headers.get("Transfer-Encoding"):
                    self.send_error(400)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > owner.max_payload_bytes:
                        self.send_error(413)
                        return
                    raw = self.rfile.read(length)
                    if len(raw) != length:
                        self.send_error(400)
                        return
                    envelope = strict_json(raw)
                    encoded = owner.auth.verify(envelope, kind="request")
                except AuthenticationError:
                    self.send_error(403, "Authentication failed")
                    return
                except (ValueError, TypeError, RecursionError):
                    self.send_error(400, "Invalid request")
                    return
                try:
                    reply = owner._dispatch(encoded, envelope["member_id"])
                except Exception as error:
                    reply = {"ok": False, "error": f"{type(error).__name__}: {str(error)[:1000]}"}
                signed = owner.auth.sign(reply, kind="response", in_reply_to=envelope["nonce"])
                body = canonical_json(signed)
                if len(body) > owner.max_payload_bytes:
                    self.send_error(413, "Response exceeds byte limit")
                    return
                # Emit a complete framed response in one write. This also avoids
                # separate tiny header/body writes on Windows loopback sockets.
                headers = ("HTTP/1.0 200 OK\r\n"
                           f"Date: {self.date_time_string()}\r\n"
                           "Content-Type: application/json\r\n"
                           f"Content-Length: {len(body)}\r\n"
                           "Connection: close\r\n\r\n").encode("ascii")
                self.close_connection = True
                self.wfile.write(headers + body)
                self.wfile.flush()

        self._server = LimitedServer((self.host, self.port), Handler)
        if self._tls_context:
            # Defer TLS to the bounded handler threads; an idle TCP client must
            # not block accept() before the concurrency/timeout limits apply.
            self._server.socket = self._tls_context.wrap_socket(
                self._server.socket, server_side=True, do_handshake_on_connect=False)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name="sakura-rpc")
        self._thread.start()
        return self

    def _execute(self, encoded: Any, member: str) -> dict[str, Any]:
        request = decode_value(encoded, max_tensor_bytes=self.max_blob_bytes)
        if not isinstance(request, dict):
            raise SerializationError("RPC request must be a dictionary")
        result = self.handler(request, member) if self.pass_member else self.handler(request)
        if not isinstance(result, dict):
            raise SerializationError("RPC handler must return a dictionary")
        encoded_result = encode_value(result, max_tensor_bytes=self.max_blob_bytes)
        data = canonical_json(encoded_result)
        if len(data) > self.max_blob_bytes:
            raise SerializationError("RPC result exceeds configured blob byte limit")
        if len(data) > self.max_payload_bytes - 2048:
            return {"ok": True, "blob": self._blobs.put(member, data)}
        return {"ok": True, "result": encoded_result}

    def _dispatch(self, encoded: Any, member: str) -> dict[str, Any]:
        kind = encoded.get("kind") if isinstance(encoded, dict) else None
        if kind == "blob_begin" and set(encoded) == {"kind", "bytes", "sha256"}:
            return {"ok": True, "transfer": self._blobs.begin(member, encoded["bytes"], encoded["sha256"])}
        if kind == "blob_append" and set(encoded) == {"kind", "id", "offset", "data"}:
            if not isinstance(encoded["data"], str) or len(encoded["data"]) > 6 * 1024 * 1024:
                raise SerializationError("Invalid blob chunk")
            try:
                chunk = base64.b64decode(encoded["data"], validate=True)
            except (ValueError, binascii.Error) as error:
                raise SerializationError("Invalid blob base64 chunk") from error
            if len(chunk) > 4 * 1024 * 1024:
                raise SerializationError("Blob chunks must not exceed 4MiB")
            self._blobs.append(member, encoded["id"], encoded["offset"], chunk)
            return {"ok": True}
        if kind == "blob_call" and set(encoded) == {"kind", "id"}:
            return self._execute(strict_json(self._blobs.take(member, encoded["id"])), member)
        if kind == "blob_read" and set(encoded) == {"kind", "id", "offset", "bytes"}:
            data = self._blobs.read(member, encoded["id"], encoded["offset"], encoded["bytes"])
            return {"ok": True, "offset": encoded["offset"], "data": base64.b64encode(data).decode("ascii")}
        if kind == "blob_drop" and set(encoded) == {"kind", "id"}:
            self._blobs.drop(member, encoded["id"])
            return {"ok": True}
        return self._execute(encoded, member)

    def close(self) -> None:
        self._closed = True
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._blobs.close()

    def __enter__(self) -> "RpcServer":
        return self.start()

    def __exit__(self, *args) -> None:
        self.close()


class RpcClient:
    """Signed RPC with transparent bounded chunk transfer in both directions.

    Large serialized trees are spooled and hashed in chunks. Deserialization still
    materializes each snapshot in CPU RAM; this is not a 35B optimizer streaming
    checkpoint engine. Increase limits only with an explicit RAM/disk budget.
    """
    def __init__(self, identity: Identity, allowed_members: Mapping[str, str], run_hash: str,
                 *, timeout: float = 30, max_payload_bytes: int = MAX_PAYLOAD_BYTES,
                 max_clock_skew: float = 120, max_blob_bytes: int = MAX_BLOB_BYTES):
        self.auth = MessageAuthenticator(identity, allowed_members, run_hash, max_clock_skew=max_clock_skew)
        if max_payload_bytes < 4096 or max_blob_bytes < max_payload_bytes:
            raise ValueError("RPC wire limit must be >=4096 bytes and <= blob limit")
        self.timeout, self.max_payload_bytes = timeout, max_payload_bytes
        self.max_blob_bytes = max_blob_bytes
        self.chunk_bytes = min(4 * 1024 * 1024, (max_payload_bytes - 2048) // 2)

    def call(self, url: str, request: dict[str, Any], *, expected_member: str) -> dict[str, Any]:
        encoded = encode_value(request, max_tensor_bytes=self.max_blob_bytes)
        data = canonical_json(encoded)
        if len(data) > self.max_blob_bytes:
            raise SerializationError("RPC request exceeds configured blob byte limit")
        if len(data) <= self.max_payload_bytes - 2048:
            reply = self._exchange(url, encoded, expected_member=expected_member)
        else:
            digest = hashlib.sha256(data).hexdigest()
            descriptor = self._exchange(url, {"kind": "blob_begin", "bytes": len(data), "sha256": digest},
                                        expected_member=expected_member)["transfer"]
            self._validate_descriptor(descriptor)
            identifier = descriptor["id"]
            try:
                for offset in range(0, len(data), self.chunk_bytes):
                    self._exchange(url, {"kind": "blob_append", "id": identifier, "offset": offset,
                                   "data": base64.b64encode(data[offset:offset + self.chunk_bytes]).decode("ascii")},
                                   expected_member=expected_member)
                del data
                reply = self._exchange(url, {"kind": "blob_call", "id": identifier}, expected_member=expected_member)
            except Exception:
                self._drop_quietly(url, identifier, expected_member)
                raise
        if "blob" in reply:
            descriptor = reply["blob"]
            self._validate_descriptor(descriptor)
            identifier = descriptor["id"]
            try:
                digest = hashlib.sha256()
                with tempfile.TemporaryFile() as spool:
                    offset = 0
                    while offset < descriptor["bytes"]:
                        part = self._exchange(url, {"kind": "blob_read", "id": identifier, "offset": offset,
                                              "bytes": self.chunk_bytes}, expected_member=expected_member)
                        if part.get("offset") != offset or not isinstance(part.get("data"), str):
                            raise RpcError("Invalid signed blob chunk")
                        try:
                            chunk = base64.b64decode(part["data"], validate=True)
                        except (ValueError, binascii.Error) as error:
                            raise RpcError("Invalid signed blob chunk encoding") from error
                        if not chunk or len(chunk) > min(self.chunk_bytes, descriptor["bytes"] - offset):
                            raise RpcError("Blob chunk length does not match descriptor")
                        spool.write(chunk)
                        digest.update(chunk)
                        offset += len(chunk)
                    if digest.hexdigest() != descriptor["sha256"]:
                        raise RpcError("Downloaded blob SHA256 verification failed")
                    spool.seek(0)
                    encoded_result = strict_json(spool.read())
            finally:
                self._drop_quietly(url, identifier, expected_member)
        else:
            encoded_result = reply["result"]
        result = decode_value(encoded_result, max_tensor_bytes=self.max_blob_bytes)
        if not isinstance(result, dict):
            raise RpcError("RPC result must be a dictionary")
        return result

    def _validate_descriptor(self, descriptor: Any) -> None:
        if (not isinstance(descriptor, dict) or set(descriptor) != {"id", "bytes", "sha256"}
                or not isinstance(descriptor["id"], str) or len(descriptor["id"]) != 48
                or type(descriptor["bytes"]) is not int or not 0 < descriptor["bytes"] <= self.max_blob_bytes
                or not isinstance(descriptor["sha256"], str) or len(descriptor["sha256"]) != 64):
            raise RpcError("Invalid or oversized signed blob descriptor")

    def _drop_quietly(self, url: str, identifier: str, expected_member: str) -> None:
        try:
            self._exchange(url, {"kind": "blob_drop", "id": identifier}, expected_member=expected_member)
        except Exception:
            pass  # Server TTL also removes transfers after disconnect/failure.

    def _exchange(self, url: str, encoded: Any, *, expected_member: str) -> dict[str, Any]:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("RPC URL must be an explicit HTTP(S) address without credentials")
        envelope = self.auth.sign(encoded)
        raw = canonical_json(envelope)
        if len(raw) > self.max_payload_bytes:
            raise SerializationError("RPC request exceeds byte limit")
        query = urllib.request.Request(url, raw, headers={"Content-Type": "application/json"}, method="POST")
        # Do not send signed training messages through ambient HTTP proxies or redirects.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        try:
            with opener.open(query, timeout=self.timeout) as response:
                reply_raw = response.read(self.max_payload_bytes + 1)
        except urllib.error.HTTPError as error:
            raise RpcError(f"RPC HTTP status {error.code}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise RpcError(f"RPC connection failed: {error}") from error
        if len(reply_raw) > self.max_payload_bytes:
            raise SerializationError("RPC response exceeds byte limit")
        reply = self.auth.verify(strict_json(reply_raw), kind="response", expected_member=expected_member,
                                 in_reply_to=envelope["nonce"])
        if not isinstance(reply, dict) or type(reply.get("ok")) is not bool:
            raise RpcError("Invalid signed RPC response")
        if not reply["ok"]:
            raise RpcError(str(reply.get("error", "Remote operation failed")))
        return reply
