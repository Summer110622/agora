import copy
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import torch

from sakura.auth import AuthenticationError, Identity, MessageAuthenticator, canonical_json, manifest_hash
from sakura.discovery import StaticDiscovery
from sakura.transport import (LocalTransport, RpcClient, RpcError, RpcServer, SerializationError,
                              decode_value, encode_value)


RUN_HASH = "a" * 64


class AuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.alice, self.bob = Identity.generate("alice"), Identity.generate("bob")
        self.members = {"alice": self.alice.public_key, "bob": self.bob.public_key}
        self.sender = MessageAuthenticator(self.alice, self.members, RUN_HASH)
        self.receiver = MessageAuthenticator(self.bob, self.members, RUN_HASH)

    def test_identity_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "private.json"
            self.alice.save(path)
            self.assertEqual(Identity.load(path).public_key, self.alice.public_key)
            with self.assertRaises(FileExistsError):
                self.alice.save(path)

    def test_unicode_manifest_uses_same_canonical_json_as_run_identity(self):
        import hashlib
        from sakura.manifest import canonical_json as manifest_json
        value = {"run_id": "さくら", "members": self.members}
        self.assertEqual(manifest_hash(value), hashlib.sha256(manifest_json(value)).hexdigest())

    def test_valid_signature_and_replay_rejected(self):
        envelope = self.sender.sign({"op": "info"})
        self.assertEqual(self.receiver.verify(envelope), {"op": "info"})
        with self.assertRaisesRegex(AuthenticationError, "Replayed"):
            self.receiver.verify(envelope)

    def test_tamper_unknown_and_expired(self):
        envelope = self.sender.sign({"op": "info"})
        modified = copy.deepcopy(envelope)
        modified["payload"]["op"] = "step"
        with self.assertRaises(AuthenticationError):
            self.receiver.verify(modified)
        unknown = Identity.generate("outsider")
        foreign = MessageAuthenticator(unknown, {"outsider": unknown.public_key}, RUN_HASH)
        with self.assertRaisesRegex(AuthenticationError, "Unknown"):
            self.receiver.verify(foreign.sign({}))
        with self.assertRaisesRegex(AuthenticationError, "timestamp"):
            self.receiver.verify(self.sender.sign({}, timestamp=time.time() - 300))

    def test_manifest_binding_and_response_binding(self):
        wrong_run = MessageAuthenticator(self.bob, self.members, "b" * 64)
        with self.assertRaises(AuthenticationError):
            wrong_run.verify(self.sender.sign({}))
        request = self.sender.sign({})
        response = self.receiver.sign({"result": 1}, kind="response", in_reply_to=request["nonce"])
        with self.assertRaises(AuthenticationError):
            self.sender.verify(response, kind="response", in_reply_to="0" * 48)
        self.assertEqual(self.sender.verify(response, kind="response", expected_member="bob",
                                            in_reply_to=request["nonce"]), {"result": 1})


class SerializationTests(unittest.TestCase):
    def test_tensor_dtypes_scalar_empty_noncontiguous(self):
        for dtype in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                      torch.float16, torch.bfloat16, torch.float32, torch.float64):
            tensor = torch.arange(6).reshape(2, 3).to(dtype).T
            result = decode_value(encode_value({1: tensor, "rng": (torch.tensor(3), torch.empty(0, 2))}))
            self.assertEqual(result[1].dtype, dtype)
            self.assertTrue(torch.equal(result[1], tensor))
            self.assertEqual(result["rng"][0].shape, torch.Size([]))
            self.assertEqual(result["rng"][1].shape, torch.Size([0, 2]))

    def test_numpy_rng_and_python_tuples(self):
        import numpy as np
        value = {"numpy": np.random.get_state(), "python": (3, (1, 2, 3), None)}
        result = decode_value(encode_value(value))
        self.assertEqual(result["numpy"][0], value["numpy"][0])
        self.assertTrue(np.array_equal(result["numpy"][1], value["numpy"][1]))
        self.assertEqual(result["python"], value["python"])

    def test_invalid_tensor_metadata_and_aggregate_limit(self):
        node = encode_value(torch.ones(2))
        node["shape"] = [3]
        with self.assertRaises(SerializationError):
            decode_value(node)
        with self.assertRaises(SerializationError):
            decode_value({"kind": "pickle", "data": "untrusted"})
        with self.assertRaises(SerializationError):
            decode_value(encode_value([torch.ones(2), torch.ones(2)]), max_tensor_bytes=12)
        with self.assertRaises(SerializationError):
            encode_value(torch.ones(100), max_tensor_bytes=16)
        with self.assertRaises(SerializationError):
            encode_value([torch.ones(2), torch.ones(2)], max_tensor_bytes=12)

    def test_local_transport_preserves_tensor_ownership(self):
        tensor = torch.ones(3)
        def handler(request):
            request["tensor"].zero_()
            return request
        result = LocalTransport(handler).call({"tensor": tensor})
        self.assertEqual(result["tensor"].sum().item(), 0)
        self.assertEqual(tensor.sum().item(), 3)


class HttpNetworkTests(unittest.TestCase):
    def setUp(self):
        self.client_identity = Identity.generate("trainer")
        self.worker_identity = Identity.generate("worker")
        self.members = {"trainer": self.client_identity.public_key, "worker": self.worker_identity.public_key}
        self.server = RpcServer(lambda request: {"output": request["input"] * 2}, self.worker_identity,
                                self.members, RUN_HASH).start()
        self.client = RpcClient(self.client_identity, self.members, RUN_HASH)

    def tearDown(self):
        self.server.close()

    def test_roundtrip_real_cpu_tensor_and_expected_member(self):
        result = self.client.call(self.server.url, {"input": torch.arange(3.)}, expected_member="worker")
        self.assertTrue(torch.equal(result["output"], torch.arange(3.) * 2))
        with self.assertRaises(AuthenticationError):
            self.client.call(self.server.url, {"input": torch.ones(1)}, expected_member="trainer")

    def test_tensor_response_larger_than_write_buffer(self):
        tensor = torch.arange(65536, dtype=torch.float32)
        client = RpcClient(self.client_identity, self.members, RUN_HASH, timeout=5)
        result = client.call(self.server.url, {"input": tensor}, expected_member="worker")
        self.assertTrue(torch.equal(result["output"], tensor * 2))

    def test_unknown_member_and_wrong_manifest_fail(self):
        stranger = Identity.generate("stranger")
        outsider = RpcClient(stranger, {"stranger": stranger.public_key, **self.members}, RUN_HASH)
        with self.assertRaisesRegex(RpcError, "403"):
            outsider.call(self.server.url, {"input": torch.ones(1)}, expected_member="worker")
        other_run = RpcClient(self.client_identity, self.members, "b" * 64)
        with self.assertRaisesRegex(RpcError, "403"):
            other_run.call(self.server.url, {"input": torch.ones(1)}, expected_member="worker")

    def test_http_replay_is_rejected(self):
        raw = canonical_json(self.client.auth.sign(encode_value({"input": torch.ones(1)})))
        request = urllib.request.Request(self.server.url, raw, method="POST")
        with urllib.request.urlopen(request) as response:
            self.assertEqual(response.status, 200)
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 403)

    def test_remote_failure_and_payload_bound(self):
        with self.assertRaisesRegex(RpcError, "KeyError"):
            self.client.call(self.server.url, {}, expected_member="worker")
        small = RpcClient(self.client_identity, self.members, RUN_HASH, max_payload_bytes=4096, max_blob_bytes=4096)
        with self.assertRaises(SerializationError):
            small.call(self.server.url, {"input": torch.ones(2048)}, expected_member="worker")

    def test_chunked_request_and_response_above_wire_limit(self):
        with RpcServer(lambda request: {"output": request["input"] * 2}, self.worker_identity, self.members,
                       RUN_HASH, max_payload_bytes=16_384, max_blob_bytes=2 * 1024 * 1024) as server:
            client = RpcClient(self.client_identity, self.members, RUN_HASH, timeout=5,
                               max_payload_bytes=16_384, max_blob_bytes=2 * 1024 * 1024)
            tensor = torch.arange(32_768, dtype=torch.float32)
            result = client.call(server.url, {"input": tensor}, expected_member="worker")
            self.assertTrue(torch.equal(result["output"], tensor * 2))
            self.assertEqual(server._blobs._records, {})

    def test_chunk_transfer_owner_and_whole_blob_hash(self):
        # Internal transfer calls are still signed and bound to the exact member.
        descriptor = self.client._exchange(self.server.url, {"kind": "blob_begin", "bytes": 1,
                                            "sha256": "0" * 64}, expected_member="worker")["transfer"]
        identifier = descriptor["id"]
        other_member = RpcClient(self.worker_identity, self.members, RUN_HASH, timeout=5)
        with self.assertRaisesRegex(RpcError, "another member"):
            other_member._exchange(self.server.url, {"kind": "blob_read", "id": identifier,
                                    "offset": 0, "bytes": 1}, expected_member="worker")
        self.client._exchange(self.server.url, {"kind": "blob_append", "id": identifier,
                              "offset": 0, "data": "eA=="}, expected_member="worker")
        with self.assertRaisesRegex(RpcError, "SHA256"):
            self.client._exchange(self.server.url, {"kind": "blob_call", "id": identifier}, expected_member="worker")
        self.client._drop_quietly(self.server.url, identifier, "worker")

    def test_corrupt_download_chunk_does_not_deserialize(self):
        tensor = torch.ones(16_384)
        with RpcServer(lambda request: {"output": tensor}, self.worker_identity, self.members,
                       RUN_HASH, max_payload_bytes=16_384, max_blob_bytes=1024 * 1024) as server:
            original_read = server._blobs.read
            def corrupted(*args):
                chunk = original_read(*args)
                return bytes([chunk[0] ^ 1]) + chunk[1:]
            client = RpcClient(self.client_identity, self.members, RUN_HASH, timeout=5,
                               max_payload_bytes=16_384, max_blob_bytes=1024 * 1024)
            with mock.patch.object(server._blobs, "read", side_effect=corrupted):
                with self.assertRaisesRegex(RpcError, "SHA256"):
                    client.call(server.url, {}, expected_member="worker")
            self.assertEqual(server._blobs._records, {})

    def test_external_bind_requires_explicit_opt_in_and_static_discovery(self):
        with self.assertRaises(ValueError):
            RpcServer(lambda x: x, self.worker_identity, self.members, RUN_HASH, host="0.0.0.0")
        discovery = StaticDiscovery({"worker": self.server.url}, allowed_members=self.members)
        self.assertEqual(discovery.resolve(), {"worker": self.server.url})
        with self.assertRaises(ValueError):
            StaticDiscovery({"stranger": self.server.url}, allowed_members=self.members)

    def test_bounded_concurrency_and_closed_server(self):
        entered, release = threading.Event(), threading.Event()
        def handler(request):
            entered.set()
            release.wait(5)
            return {}
        with RpcServer(handler, self.worker_identity, self.members, RUN_HASH,
                       max_concurrent_requests=1) as server:
            outcomes = []
            client = RpcClient(self.client_identity, self.members, RUN_HASH, timeout=5)
            def first_request():
                try:
                    outcomes.append(client.call(server.url, {}, expected_member="worker"))
                except Exception as error:
                    outcomes.append(error)
            thread = threading.Thread(target=first_request)
            thread.start()
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaises(RpcError):
                    client.call(server.url, {}, expected_member="worker")
            finally:
                release.set()
                thread.join(6)
            self.assertEqual(outcomes, [{}])
        with self.assertRaises(RuntimeError):
            server.start()


if __name__ == "__main__":
    unittest.main()
