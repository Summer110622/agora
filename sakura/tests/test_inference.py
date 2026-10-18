import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from sakura.inference import CachedInferenceStage, CachedStageWorker, CachedPipeline, resolve_inference_dtype
from sakura.model import LoRALinear, build_model, partition_model, tiny_config
from sakura.trainer import StageEndpoint
from sakura.transport import LocalTransport


def fixture_config():
    return tiny_config(vocab_size=67, hidden_size=16, num_hidden_layers=4,
                       num_attention_heads=2, num_key_value_heads=1,
                       linear_num_key_heads=1, linear_num_value_heads=2,
                       moe_intermediate_size=8, shared_expert_intermediate_size=8, num_experts=4)


def fixture_manifest(stages=2):
    return {"schema_version": 1, "run_id": "cached-inference-fixture", "mode": "lora",
            "model": {"config": fixture_config(), "seed": 5, "revision": "random-5"},
            "tokenizer": {"kind": "byte", "revision": "fixture-utf8-v1"},
            "dataset": {"revision": "fixture-only"}, "num_stages": stages, "members": {},
            "training": {"learning_rate": 0.001, "max_grad_norm": 1.0,
                         "sequence_length": 8, "checkpoint_every": 1}}


class CachedInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.ids = torch.tensor([[1, 5, 6, 7, 8]])

    def workers(self, count=2, model=None, **options):
        manifest = fixture_manifest(count)
        source = model or build_model(fixture_config(), mode="full", seed=5)
        stages = partition_model(copy.deepcopy(source), count)
        workers = [CachedStageWorker(manifest, index, stage=stage, **options)
                   for index, stage in enumerate(stages)]
        return workers

    def pipeline(self, workers):
        endpoints = [StageEndpoint(index, LocalTransport(worker.handle).call)
                     for index, worker in enumerate(workers)]
        return CachedPipeline(endpoints)

    def reference(self, model, count):
        ids, generated = self.ids.clone(), []
        model.eval()
        with torch.inference_mode():
            for _ in range(count):
                token = model(ids, use_cache=False, output_router_logits=False).logits[:, -1].argmax(-1)
                generated.append(int(token))
                ids = torch.cat((ids, token[:, None]), dim=1)
        return torch.tensor([generated])

    def test_actual_qwen_uncached_single_stage_and_two_stage_greedy_match(self):
        model = build_model(fixture_config(), mode="full", seed=5)
        expected = self.reference(model, 6)
        for count in (1, 2):
            workers = self.workers(count, model)
            result = self.pipeline(workers).generate(self.ids, 6)
            torch.testing.assert_close(result["tokens"], expected)
            torch.testing.assert_close(result["output_ids"], torch.cat((self.ids, expected), dim=1))
            self.assertEqual(result["metrics"]["generated_tokens"], 6)
            self.assertGreater(result["metrics"]["prefill_seconds"], 0)
            self.assertGreater(result["metrics"]["decode_seconds"], 0)
            self.assertEqual(result["metrics"]["cleanup_failures"], 0)
            self.assertTrue(all(not worker.sessions for worker in workers))

    def test_decode_processes_only_new_token_and_updates_hybrid_cache(self):
        workers = self.workers(2)
        calls, handles = [], []
        for worker in workers:
            for layer in worker.stage.layers:
                handles.append(layer.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[1])))
        try:
            head = workers[0].handle({"op": "prefill", "session_id": "cache-test", "inputs": self.ids, "position": 0})
            tail = workers[1].handle({"op": "prefill", "session_id": "cache-test",
                                      "inputs": head["hidden_states"], "position": 0})
            self.assertNotIn("logits", tail)
            self.assertNotIn("hidden_states", tail)
            self.assertEqual(calls, [5, 5, 5, 5])
            old_state = workers[0].sessions["cache-test"].cache.layers[0].recurrent_states.clone()
            cache = workers[1].sessions["cache-test"].cache
            self.assertEqual(cache.get_seq_length(), 5)
            self.assertGreater(tail["cache_bytes"], 0)
            head = workers[0].handle({"op": "decode", "session_id": "cache-test",
                                      "inputs": tail["next_token"].reshape(1, 1), "position": 5})
            tail = workers[1].handle({"op": "decode", "session_id": "cache-test",
                                      "inputs": head["hidden_states"], "position": 5})
            self.assertEqual(calls[4:], [1, 1, 1, 1])
            self.assertEqual(cache.get_seq_length(), 6)
            self.assertFalse(torch.equal(old_state, workers[0].sessions["cache-test"].cache.layers[0].recurrent_states))
            self.assertEqual(head["processed_tokens"], 1)
            self.assertEqual(tail["processed_tokens"], 1)
            self.assertTrue(all(not p.requires_grad and p.grad is None for worker in workers for p in worker.stage.parameters()))
        finally:
            for handle in handles:
                handle.remove()

    def test_lora_stage_is_merged_frozen_and_matches_merged_reference(self):
        from sakura.model import merge_lora
        adapted = build_model(fixture_config(), mode="lora", seed=5)
        with torch.no_grad():
            for name, parameter in adapted.named_parameters():
                if name.endswith("lora_B"):
                    parameter.fill_(0.02)
        workers = self.workers(2, adapted)
        self.assertFalse(any(isinstance(module, LoRALinear) for worker in workers for module in worker.stage.modules()))
        expected = self.reference(merge_lora(adapted), 3)
        actual = self.pipeline(workers).generate(self.ids, 3)["tokens"]
        torch.testing.assert_close(actual, expected)

    def test_exported_adapter_load_parity_and_mixed_adapter_identity_rejected(self):
        from sakura.model import export_lora_checkpoint, merge_lora
        adapted = build_model(fixture_config(), mode="lora", seed=5)
        with torch.no_grad():
            for name, parameter in adapted.named_parameters():
                if name.endswith("lora_B"):
                    parameter.fill_(0.02)
        states = {str(index): stage.state_dict() for index, stage in enumerate(
            partition_model(copy.deepcopy(adapted), 2))}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            export_lora_checkpoint(states, fixture_config(), 2, directory, "fixture", "random-5")
            workers = self.workers(2, adapter_path=directory)
            torch.testing.assert_close(self.pipeline(workers).generate(self.ids, 3)["tokens"],
                                       self.reference(merge_lora(adapted), 3))
            self.assertIsNotNone(workers[0].handle({"op": "info"})["adapter_sha256"])
            # A harmless metadata difference still identifies a different export.
            options_path = directory / "adapter_config.json"
            options = json.loads(options_path.read_text(encoding="utf-8"))
            options["base_model_name_or_path"] = "another-export"
            options_path.write_text(json.dumps(options), encoding="utf-8")
            other = self.workers(2, adapter_path=directory)
            with self.assertRaisesRegex(ValueError, "different adapters"):
                self.pipeline([workers[0], other[1]])

    def test_session_limits_positions_ttl_reset_and_context_are_bounded(self):
        worker = self.workers(1, max_sessions=1, max_context=6)[0]
        request = {"op": "prefill", "session_id": "one", "inputs": self.ids, "position": 0}
        reply = worker.handle(request)
        with self.assertRaisesRegex(ValueError, "Prefill"):
            worker.handle(request)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            worker.handle({**request, "session_id": "two"})
        with self.assertRaisesRegex(ValueError, "position"):
            worker.handle({"op": "decode", "session_id": "one", "inputs": torch.tensor([[1]]), "position": 0})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            worker.handle({"op": "decode", "session_id": "one", "inputs": self.ids, "position": 5})
        worker.handle({"op": "decode", "session_id": "one", "inputs": reply["next_token"].reshape(1, 1), "position": 5})
        with self.assertRaisesRegex(ValueError, "context"):
            worker.handle({"op": "decode", "session_id": "one", "inputs": torch.tensor([[1]]), "position": 6})
        self.assertEqual(worker.sessions["one"].position, 6)
        worker.sessions["one"].expires_at = 0
        with self.assertRaisesRegex(ValueError, "expired"):
            worker.handle({"op": "decode", "session_id": "one", "inputs": torch.tensor([[1]]), "position": 6})
        self.assertEqual(worker.handle({"op": "info"})["sessions"], 0)
        worker.handle({**request, "session_id": "two"})
        self.assertTrue(worker.handle({"op": "reset", "session_id": "two"})["reset"])
        worker.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            worker.handle({"op": "info"})

    def test_cache_budget_rejects_before_prefill_and_cpu_dtype_defaults(self):
        worker = self.workers(1, max_cache_bytes=1)[0]
        with self.assertRaisesRegex(MemoryError, "cache"):
            worker.handle({"op": "prefill", "session_id": "too-large", "inputs": self.ids, "position": 0})
        self.assertFalse(worker.sessions)
        self.assertEqual(resolve_inference_dtype("auto", "cpu"), torch.float32)
        with self.assertRaises(ValueError):
            resolve_inference_dtype("float64", "cpu")

    def test_pipeline_stops_on_eos_and_cleans_partial_failure(self):
        workers = self.workers(2)
        pipeline = self.pipeline(workers)
        first = int(pipeline.generate(self.ids, 1)["tokens"][0, 0])
        reply = pipeline.generate(self.ids, 6, eos_token_id=first)
        self.assertEqual(reply["tokens"].shape, (1, 1))
        self.assertTrue(all(not worker.sessions for worker in workers))
        endpoints = [StageEndpoint(0, LocalTransport(workers[0].handle).call),
                     StageEndpoint(1, LocalTransport(workers[1].handle).call)]
        original = endpoints[1].call
        def fail_decode(request):
            if request["op"] == "decode":
                raise RuntimeError("fixture disconnect")
            return original(request)
        endpoints[1].call = fail_decode
        with self.assertRaisesRegex(RuntimeError, "disconnect"):
            CachedPipeline(endpoints).generate(self.ids, 3)
        self.assertTrue(all(not worker.sessions for worker in workers))

    def test_one_token_prefill_and_decode_match_reference(self):
        self.ids = self.ids[:, :1]
        model = build_model(fixture_config(), mode="full", seed=5)
        expected = self.reference(model, 4)
        observed = self.pipeline(self.workers(2, model)).generate(self.ids, 4)["tokens"]
        torch.testing.assert_close(observed, expected)


if __name__ == "__main__":
    unittest.main()
