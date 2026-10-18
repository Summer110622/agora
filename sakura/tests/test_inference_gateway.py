"""Reverse-poll authorization, queue lifecycle, and real cached dense generation."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import ipaddress
import math
from pathlib import Path
import socket
import ssl
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock
import urllib.request

import torch

from sakura.auth import Identity, canonical_json, strict_json
from sakura.inference_gateway import (InferenceGateway, process_one_job, request_generation,
                                      validate_endpoint, validate_generation)
from sakura.inference_generate import HFGenerator, inference_dtype
from sakura.transport import RpcClient, RpcError, RpcServer, decode_value, encode_value


RUN_HASH = "d" * 64


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class GatewayLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.gateway = InferenceGateway("pc", "gpu", max_jobs=2, job_timeout=5, result_ttl=7,
                                        clock=self.clock)

    def submit(self, number=1, prompt="日本語で加算関数", tokens=4):
        identifier = f"job{number:016d}"
        reply = self.gateway.handle({"op": "submit", "job_id": identifier,
                                     "prompt": prompt, "max_new_tokens": tokens}, "pc")
        return identifier, reply

    def take(self):
        return self.gateway.handle({"op": "take", "wait_seconds": 0}, "gpu")["job"]

    def complete(self, job, **fields):
        return self.gateway.handle({"op": "complete", "job_id": job["job_id"],
                                    "lease": job["lease"], **fields}, "gpu")

    def get(self, identifier):
        return self.gateway.handle({"op": "get", "job_id": identifier}, "pc")

    def test_queue_bound_submission_idempotence_and_conflicting_prompt(self):
        first, reply = self.submit(1)
        self.assertEqual(reply["status"], "queued")
        self.submit(2)
        self.assertEqual(self.submit(1)[1]["job_id"], first)
        with self.assertRaisesRegex(ValueError, "another prompt"):
            self.submit(1, prompt="different")
        with self.assertRaisesRegex(RuntimeError, "full"):
            self.submit(3)
        self.assertEqual(self.gateway.handle({"op": "info"}, "pc")["jobs"], 2)

    def test_leases_serialize_workers_and_completion_is_idempotent(self):
        first, _ = self.submit(1)
        second, _ = self.submit(2)
        job = self.take()
        self.assertEqual(job["job_id"], first)
        self.assertEqual(len(job["lease"]), 64)
        self.assertIsNone(self.take())
        with self.assertRaises(PermissionError):
            self.gateway.handle({"op": "complete", "job_id": first, "lease": "wrong",
                                 "result": {"text": "bad"}}, "gpu")
        result = {"text": "def add(a,b): return a+b", "metrics": {"generated_tokens": 8}}
        self.assertEqual(self.complete(job, result=result), {"accepted": True})
        self.assertEqual(self.complete(job, result=copy.deepcopy(result)), {"accepted": True})
        with self.assertRaisesRegex(ValueError, "already finished"):
            self.complete(job, result={"text": "different"})
        self.assertEqual(self.get(first), {"status": "done", "result": result})
        self.assertEqual(self.take()["job_id"], second)
        # Completed results are also bounded queue entries until their TTL.
        with self.assertRaisesRegex(RuntimeError, "full"):
            self.submit(3)

    def test_deadline_fails_running_job_rejects_late_completion_then_expires(self):
        identifier, _ = self.submit()
        job = self.take()
        self.clock.advance(5)
        self.assertEqual(self.get(identifier)["status"], "failed")
        self.assertIn("deadline", self.get(identifier)["error"])
        with self.assertRaisesRegex(ValueError, "finished or expired"):
            self.complete(job, result={"text": "too late"})
        self.clock.advance(7)
        with self.assertRaises(KeyError):
            self.get(identifier)
        self.submit()
        self.assertNotEqual(self.take()["lease"], job["lease"])

    def test_queued_deadline_failure_and_restart_do_not_replay_jobs(self):
        identifier, _ = self.submit()
        self.clock.advance(6)
        self.assertIsNone(self.take())
        self.assertEqual(self.get(identifier)["status"], "failed")
        restarted = InferenceGateway("pc", "gpu", clock=self.clock)
        with self.assertRaises(KeyError):
            restarted.handle({"op": "get", "job_id": identifier}, "pc")
        self.assertIsNone(restarted.handle({"op": "take", "wait_seconds": 0}, "gpu")["job"])

    def test_failure_completion_and_validation_boundaries(self):
        identifier, _ = self.submit()
        job = self.take()
        for fields in ({}, {"result": {"text": "a"}, "error": "b"},
                       {"result": {"tokens": []}}, {"error": "x" * 2049},
                       {"result": {"text": "x" * (1024 * 1024 + 1)}}):
            with self.assertRaises(ValueError):
                self.complete(job, **fields)
        self.complete(job, error="OutOfMemoryError: retry with a smaller model")
        self.assertEqual(self.get(identifier)["status"], "failed")
        self.assertEqual(self.complete(job, error="OutOfMemoryError: retry with a smaller model"), {"accepted": True})
        for wait in (-1, 21, True, float("nan")):
            with self.assertRaises(ValueError):
                self.gateway.handle({"op": "take", "wait_seconds": wait}, "gpu")

    def test_roles_cannot_be_selected_by_request_fields(self):
        for member, request in (("pc", {"op": "take", "wait_seconds": 0, "member": "gpu"}),
                                ("gpu", {"op": "info", "member": "pc"}),
                                ("other", {"op": "info", "member": "pc"})):
            with self.assertRaises(PermissionError):
                self.gateway.handle(request, member)

    def test_generate_once_even_when_completion_ack_is_lost_twice(self):
        identifier, _ = self.submit()
        generator = mock.Mock()
        generator.generate.return_value = {"text": "result"}
        completions = []
        def call(request):
            reply = self.gateway.handle(request, "gpu")
            if request["op"] == "complete":
                completions.append(copy.deepcopy(request))
                if len(completions) < 3:
                    raise RpcError("Acknowledgement was lost after accepting completion")
            return reply
        with mock.patch("sakura.inference_gateway.time.sleep"):
            self.assertTrue(process_one_job(call, generator, wait_seconds=0))
        generator.generate.assert_called_once_with("日本語で加算関数", max_new_tokens=4)
        self.assertEqual(len(completions), 3)
        self.assertEqual(completions[0], completions[2])
        self.assertEqual(self.get(identifier)["result"], {"text": "result"})
        self.assertFalse(process_one_job(call, generator, wait_seconds=0))
        self.assertEqual(generator.generate.call_count, 1)

    def test_generator_error_is_reported_without_regeneration(self):
        identifier, _ = self.submit()
        generator = mock.Mock()
        generator.generate.side_effect = ValueError("context too long")
        self.assertTrue(process_one_job(lambda request: self.gateway.handle(request, "gpu"),
                                        generator, wait_seconds=0))
        generator.generate.assert_called_once()
        self.assertEqual(self.get(identifier)["error"], "ValueError: context too long")

    def test_request_result_failure_and_timeout_do_not_resubmit(self):
        calls = []
        def completed(request):
            calls.append(request)
            return {"status": "queued"} if request["op"] == "submit" else {
                "status": "done", "result": {"text": "done"}}
        with mock.patch("sakura.inference_gateway.time.perf_counter", side_effect=[100, 100.125]):
            result = request_generation(completed, "code", max_new_tokens=2)
        self.assertEqual(result["text"], "done")
        self.assertEqual(result["end_to_end_seconds"], .125)
        self.assertEqual(sum(request["op"] == "submit" for request in calls), 1)
        with self.assertRaisesRegex(RuntimeError, "worker failure"):
            request_generation(lambda request: {"status": "failed", "error": "worker failure"}, "code")
        calls.clear()
        def pending(request):
            calls.append(request)
            return {"status": "queued"}
        with mock.patch("sakura.inference_gateway.time.perf_counter", side_effect=[0, 2]):
            with self.assertRaises(TimeoutError):
                request_generation(pending, "code", timeout=1)
        self.assertEqual(sum(request["op"] == "submit" for request in calls), 1)

    def test_remote_url_and_prompt_validation(self):
        for url in ("http://example.com/rpc", "https://x/not-rpc", "https://u:p@x/rpc",
                    "https://x/rpc?q=1", "https://x/rpc#fragment"):
            with self.assertRaises(ValueError):
                validate_endpoint(url)
        self.assertEqual(validate_endpoint("https://example.com/rpc"), "https://example.com/rpc")
        self.assertEqual(validate_endpoint("http://127.0.0.1:123/rpc"), "http://127.0.0.1:123/rpc")
        validate_endpoint("http://private-host/rpc", allow_insecure_http=True)
        for prompt, count in ((" ", 1), ("a" * 65537, 1), ("a", True), ("a", 0), ("a", 2049)):
            with self.assertRaises(ValueError):
                validate_generation(prompt, count)


class SignedGatewayTests(unittest.TestCase):
    def setUp(self):
        self.identities = {name: Identity.generate(name) for name in ("pc", "gpu", "gateway", "observer")}
        self.members = {name: identity.public_key for name, identity in self.identities.items()}
        self.gateway = InferenceGateway("pc", "gpu")

    def client(self, name):
        return RpcClient(self.identities[name], self.members, RUN_HASH, timeout=5)

    def assert_host_tls_supported(self, cert_path, key_path):
        """Separate host TLS failures from the application regression below.

        Some managed Windows hosts drop even a minimal Python TLS echo reply.
        This independent preflight uses the same trusted certificate but none
        of Sakura's listener, framing, authentication, or handler code.
        """
        server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_context.load_cert_chain(str(cert_path), str(key_path))
        client_context = ssl.create_default_context(cafile=str(cert_path))
        server_errors = []
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(3)

            def echo():
                try:
                    connection, _ = listener.accept()
                    connection.settimeout(3)
                    with server_context.wrap_socket(connection, server_side=True) as secure:
                        if secure.recv(32) != b"tls-preflight":
                            raise AssertionError("Unexpected TLS preflight input")
                        secure.sendall(b"tls-preflight-ok")
                except Exception as error:
                    server_errors.append(error)

            thread = threading.Thread(target=echo, daemon=True)
            thread.start()
            try:
                with socket.create_connection(listener.getsockname(), timeout=3) as connection:
                    with client_context.wrap_socket(connection, server_hostname="127.0.0.1") as secure:
                        secure.sendall(b"tls-preflight")
                        reply = secure.recv(32)
            except (ssl.SSLError, TimeoutError) as error:
                self.skipTest(f"Host Python TLS loopback echo preflight failed: {type(error).__name__}: {error}")
            finally:
                thread.join(timeout=4)
        self.assertFalse(thread.is_alive(), "Minimal TLS echo did not finish")
        self.assertEqual(server_errors, [])
        self.assertEqual(reply, b"tls-preflight-ok")

    def test_verified_signer_authorization_over_real_http(self):
        with RpcServer(self.gateway.handle, self.identities["gateway"], self.members, RUN_HASH,
                       pass_member=True) as server:
            pc, gpu = self.client("pc"), self.client("gpu")
            identifier = "signedjob000000001"
            pc.call(server.url, {"op": "submit", "job_id": identifier, "prompt": "hello",
                                 "max_new_tokens": 2, "member": "gpu"}, expected_member="gateway")
            for client, request in ((pc, {"op": "take", "wait_seconds": 0, "member": "gpu"}),
                                    (gpu, {"op": "get", "job_id": identifier, "member": "pc"}),
                                    (self.client("observer"), {"op": "info", "member": "pc"})):
                with self.assertRaisesRegex(RpcError, "Authenticated member"):
                    client.call(server.url, request, expected_member="gateway")
            job = gpu.call(server.url, {"op": "take", "wait_seconds": 0}, expected_member="gateway")["job"]
            complete = {"op": "complete", "job_id": identifier, "lease": job["lease"], "result": {"text": "ok"}}
            with self.assertRaisesRegex(RpcError, "Authenticated member"):
                pc.call(server.url, {**complete, "member": "gpu"}, expected_member="gateway")
            gpu.call(server.url, complete, expected_member="gateway")
            result = pc.call(server.url, {"op": "get", "job_id": identifier}, expected_member="gateway")
            self.assertEqual(result, {"status": "done", "result": {"text": "ok"}})

    def test_tls_requires_both_files_and_untrusted_certificate_is_rejected(self):
        with self.assertRaises(ValueError):
            RpcServer(self.gateway.handle, self.identities["gateway"], self.members, RUN_HASH,
                      pass_member=True, tls_certfile="certificate.pem")
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
        now = datetime.now(timezone.utc)
        certificate = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
                       .serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(minutes=1))
                       .not_valid_after(now + timedelta(days=1))
                       .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost"),
                           x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False).sign(key, hashes.SHA256()))
        with tempfile.TemporaryDirectory() as directory:
            cert_path, key_path = Path(directory) / "cert.pem", Path(directory) / "key.pem"
            cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
            key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                  serialization.NoEncryption()))
            handler = mock.Mock(wraps=self.gateway.handle)
            with RpcServer(handler, self.identities["gateway"], self.members, RUN_HASH, pass_member=True,
                           tls_certfile=str(cert_path), tls_keyfile=str(key_path)) as server:
                self.assertTrue(server.url.startswith("https://"))
                with self.assertRaises(RpcError):
                    self.client("pc").call(server.url, {"op": "info"}, expected_member="gateway")
            handler.assert_not_called()

    def test_slow_tls_handshake_does_not_block_authenticated_connections(self):
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
        now = datetime.now(timezone.utc)
        certificate = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
                       .serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(minutes=1))
                       .not_valid_after(now + timedelta(days=1))
                       .add_extension(x509.SubjectAlternativeName([
                           x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False).sign(key, hashes.SHA256()))
        with tempfile.TemporaryDirectory() as directory:
            cert_path, key_path = Path(directory) / "cert.pem", Path(directory) / "key.pem"
            cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
            key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                  serialization.NoEncryption()))
            self.assert_host_tls_supported(cert_path, key_path)
            with RpcServer(self.gateway.handle, self.identities["gateway"], self.members, RUN_HASH,
                           pass_member=True, tls_certfile=str(cert_path), tls_keyfile=str(key_path)) as server:
                # The first connection deliberately never sends a ClientHello.
                # It must consume only its bounded handler slot, not block accept.
                idle = socket.create_connection(server._server.server_address, timeout=2)
                try:
                    client = self.client("pc")
                    envelope = client.auth.sign(encode_value({"op": "info"}))
                    query = urllib.request.Request(server.url, canonical_json(envelope),
                                                   headers={"Content-Type": "application/json"}, method="POST")
                    # Trust this exact ephemeral test CA while retaining hostname
                    # verification; do not globally disable TLS verification.
                    context = ssl.create_default_context(cafile=str(cert_path))
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
                                                          urllib.request.HTTPSHandler(context=context))
                    with opener.open(query, timeout=2) as response:
                        signed = strict_json(response.read(8192))
                    reply = client.auth.verify(signed, kind="response", expected_member="gateway",
                                               in_reply_to=envelope["nonce"])
                    self.assertTrue(reply["ok"])
                    self.assertEqual(decode_value(reply["result"])["service"], "sakura-reverse-inference")
                finally:
                    idle.close()


class TinyTokenizer:
    chat_template = None
    pad_token_id = 0
    eos_token_id = None

    def encode(self, prompt, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + [3 + value % 50 for value in prompt.encode("utf-8")]

    def decode(self, tokens, **kwargs):
        return " ".join(map(str, tokens))


def dense_fixture():
    # Real dense HF Qwen3.5 hybrid, not an MoE checkpoint coerced into a dense class.
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    config = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=8, num_key_value_heads=2, head_dim=8,
        linear_num_key_heads=4, linear_num_value_heads=8,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
        max_position_embeddings=64, attention_dropout=0.0, tie_word_embeddings=False,
        pad_token_id=0, bos_token_id=1, eos_token_id=None, use_cache=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0,
                         "partial_rotary_factor": .25, "mrope_section": [1, 0, 0], "mrope_interleaved": True})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(718)
        model = Qwen3_5ForCausalLM(config)
    model.config._attn_implementation = "eager"
    return model


class CachedDenseGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.model, self.tokenizer = dense_fixture(), TinyTokenizer()

    def test_real_dense_cached_uncached_greedy_tokens_match_and_no_training_state(self):
        before = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        with mock.patch("torch.optim.AdamW", side_effect=AssertionError("No inference optimizer")), \
             mock.patch("torch.optim.Adam", side_effect=AssertionError("No inference optimizer")):
            generator = HFGenerator(self.model, self.tokenizer, max_context=32)
            cached = generator.generate("abc", max_new_tokens=5, use_cache=True)
            uncached = generator.generate("abc", max_new_tokens=5, use_cache=False)
        self.assertEqual(cached["token_ids"], uncached["token_ids"])
        self.assertEqual(len(cached["token_ids"]), 5)
        self.assertFalse(self.model.training)
        self.assertFalse(any(parameter.requires_grad for parameter in self.model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]))
        metrics = cached["metrics"]
        self.assertEqual(metrics["prompt_tokens"], 4)
        self.assertEqual(metrics["generated_tokens"], 5)
        self.assertTrue(math.isfinite(metrics["generation_seconds"]))
        self.assertGreater(metrics["generation_seconds"], 0)
        self.assertAlmostEqual(metrics["tokens_per_second"], 5 / metrics["generation_seconds"])
        self.assertTrue(metrics["use_cache"])
        self.assertFalse(uncached["metrics"]["use_cache"])
        self.assertFalse(metrics["includes_model_load"])
        self.assertEqual(metrics["device"], "cpu")
        self.assertEqual(metrics["dtype"], "torch.float32")

    def test_context_limit_is_exact_and_never_truncates_or_calls_model_on_overflow(self):
        generator = HFGenerator(self.model, self.tokenizer, max_context=8)
        self.assertEqual(len(generator.generate("abc", max_new_tokens=4)["token_ids"]), 4)
        with mock.patch.object(self.model, "generate", wraps=self.model.generate) as generate:
            with self.assertRaisesRegex(ValueError, "context limit"):
                generator.generate("abc", max_new_tokens=5)
            generate.assert_not_called()
        self.assertEqual(HFGenerator(self.model, self.tokenizer, max_context=100).max_context, 64)

    def test_measured_benchmark_checks_same_tokens_and_reports_samples(self):
        result = HFGenerator(self.model, self.tokenizer, max_context=32).benchmark("ab", max_new_tokens=3, repeats=2)
        self.assertTrue(result["greedy_tokens_equal"])
        self.assertEqual(len(result["cached_seconds"]), 2)
        self.assertEqual(len(result["uncached_seconds"]), 2)
        self.assertEqual(result["generated_tokens"], 3)
        self.assertGreater(result["measured_cache_speedup"], 0)
        self.assertIn("no WAN", result["scope"])

    def test_hf_loader_routes_actual_vision_architecture_and_pins_same_revision(self):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
        revision = "e" * 40
        for vision in (False, True):
            with self.subTest(vision=vision):
                config = SimpleNamespace(vision_config=object()) if vision else SimpleNamespace()
                with mock.patch.object(AutoConfig, "from_pretrained", return_value=config) as load_config, \
                     mock.patch.object(AutoTokenizer, "from_pretrained", return_value=self.tokenizer) as load_tokenizer, \
                     mock.patch.object(AutoModelForCausalLM, "from_pretrained", return_value=self.model) as dense_loader, \
                     mock.patch.object(AutoModelForImageTextToText, "from_pretrained", return_value=self.model) as vision_loader:
                    generator = HFGenerator.from_pretrained("Qwen/offline-fixture", revision, device="cpu")
                    loader, unused = (vision_loader, dense_loader) if vision else (dense_loader, vision_loader)
                    loader.assert_called_once()
                    unused.assert_not_called()
                    self.assertEqual(loader.call_args.kwargs["revision"], revision)
                    self.assertEqual(loader.call_args.kwargs["dtype"], torch.float32)
                    self.assertFalse(loader.call_args.kwargs["trust_remote_code"])
                    self.assertTrue(loader.call_args.kwargs["use_safetensors"])
                    self.assertEqual(load_config.call_args.kwargs["revision"], revision)
                    self.assertEqual(load_tokenizer.call_args.kwargs["revision"], revision)
                    self.assertIs(generator.model, self.model)
        for revision in ("main", "f" * 39, "g" * 40):
            with self.assertRaises(ValueError):
                HFGenerator.from_pretrained("unused-no-network", revision)

    def test_portable_dtype_guards(self):
        self.assertEqual(inference_dtype("cpu"), torch.float32)
        self.assertEqual(inference_dtype("mps"), torch.float16)
        for device, dtype in (("cpu", "float16"), ("cpu", "bfloat16"), ("mps", "bfloat16"), ("cpu", "unknown")):
            with self.assertRaises(ValueError):
                inference_dtype(device, dtype)


if __name__ == "__main__":
    unittest.main()
