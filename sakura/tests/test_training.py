import copy
import tempfile
import unittest
from pathlib import Path

import torch

from sakura.auth import Identity
from sakura.checkpoint import CheckpointManager
from sakura.doctor import place_stages, resolve_device
from sakura.manifest import run_hash, validate_manifest
from sakura.model import build_model, tiny_config
from sakura.runtime import StageWorker
from sakura.trainer import PipelineTrainer, StageEndpoint
from sakura.transport import LocalTransport, RpcClient, RpcServer


def manifest(mode="lora"):
    return {"schema_version": 1, "run_id": "integration", "mode": mode,
            "model": {"config": tiny_config(vocab_size=64, hidden_size=32), "seed": 42, "revision": "random-42"},
            "tokenizer": {"kind": "byte", "revision": "fixture"}, "dataset": {"revision": "fixture-only"},
            "num_stages": 2, "members": {},
            "training": {"learning_rate": 0.001, "max_grad_norm": 0.02, "sequence_length": 8,
                         "checkpoint_every": 1}}


def batch():
    ids = torch.tensor([[1, 5, 6, 7, 8, 2, 0, 0]])
    labels = ids.clone()
    labels[:, :2] = -100
    labels[:, -2:] = -100
    return {"input_ids": ids, "labels": labels, "attention_mask": (ids != 0).long()}


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = manifest()

    def tearDown(self):
        self.temp.cleanup()

    def trainer(self, config=None, calls=None):
        config = config or self.manifest
        workers = [StageWorker(config, i) for i in range(2)]
        endpoints = [StageEndpoint(i, LocalTransport(w.handle).call) for i, w in enumerate(workers)]
        trainer = PipelineTrainer(config, calls or endpoints, CheckpointManager(self.root / "checkpoints"))
        return trainer, workers

    def test_distributed_three_step_optimizer_matches_monolithic_and_frozen_base(self):
        reference = build_model(self.manifest["model"]["config"], seed=42)
        initial = {n: p.detach().clone() for n, p in reference.named_parameters()}
        parameters = [p for p in reference.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=0.001)
        trainer, workers = self.trainer()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            result = reference(**batch(), use_cache=False, output_router_logits=True)
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 0.02)
            optimizer.step()
            observed = trainer.train_batch(batch())
            self.assertAlmostEqual(observed["loss"], float(result.loss.detach()), places=5)
        changed = 0
        reference_parameters = dict(reference.named_parameters())
        for worker in workers:
            for name, value in worker.stage.named_parameters():
                full = worker.stage.global_name(name)
                torch.testing.assert_close(value, reference_parameters[full], atol=2e-6, rtol=2e-5)
                if not value.requires_grad:
                    self.assertTrue(torch.equal(value, initial[full]))
                else:
                    changed += int(not torch.equal(value, initial[full]))
        self.assertGreater(changed, 0)

    def test_resume_matches_uninterrupted_model_optimizer_and_data_cursor(self):
        trainer, original = self.trainer()
        trainer.train_batch(batch())
        trainer.train_batch(batch())
        resumed, fresh = self.trainer()
        resumed.resume()
        self.assertEqual(resumed.cursor, 2)
        result = resumed.train_batch(batch())
        expected = trainer.train_batch(batch())
        self.assertAlmostEqual(result["loss"], expected["loss"], places=7)
        for a, b in zip(original, fresh):
            for key, value in a.stage.state_dict().items():
                torch.testing.assert_close(value, b.stage.state_dict()[key], rtol=0, atol=0)

    def test_partial_optimizer_commit_rolls_back_every_stage(self):
        trainer, workers = self.trainer()
        trainer.train_batch(batch())
        before = [w.snapshot() for w in workers]
        original_call = trainer.endpoints[1].call
        def fail_second_commit(request):
            if request["op"] == "step" and request["step"] == 2:
                raise ConnectionError("injected worker disconnect")
            return original_call(request)
        trainer.endpoints[1].call = fail_second_commit
        with self.assertRaisesRegex(ConnectionError, "injected"):
            trainer.train_batch(batch())
        self.assertEqual(trainer.step_number, 1)
        self.assertEqual(trainer.cursor, 1)
        for expected, worker in zip(before, workers):
            self.assertEqual(worker.step_number, 1)
            for key, tensor in expected["model"].items():
                torch.testing.assert_close(tensor, worker.stage.state_dict()[key], atol=0, rtol=0)
        trainer.endpoints[1].call = original_call
        self.assertEqual(trainer.train_batch(batch())["step"], 2)

    def test_full_weight_mode_updates_router_and_resumes(self):
        config = manifest("full")
        trainer, workers = self.trainer(config)
        router = next((n, p.detach().clone()) for n, p in workers[0].stage.named_parameters() if "mlp.gate.weight" in n)
        trainer.train_batch(batch())
        self.assertFalse(torch.equal(router[1], dict(workers[0].stage.named_parameters())[router[0]]))
        trainer.resume()
        self.assertEqual(trainer.train_batch(batch())["step"], 2)

    def test_stale_and_duplicate_commit_do_not_repeat_optimizer_update(self):
        trainer, workers = self.trainer()
        trainer.train_batch(batch())
        state = copy.deepcopy(workers[0].stage.state_dict())
        self.assertTrue(workers[0].handle({"op": "step", "step": 1})["already_applied"])
        for key, tensor in state.items():
            self.assertTrue(torch.equal(tensor, workers[0].stage.state_dict()[key]))
        with self.assertRaisesRegex(ValueError, "Stale"):
            workers[0].handle({"op": "forward", "step": 0})

    def test_trainer_restart_restores_surviving_and_pending_workers(self):
        trainer, workers = self.trainer()
        trainer.train_batch(batch())
        # The trainer disappears after head forward; workers are not restarted.
        workers[0].handle({"op": "forward", "step": 2, "inputs": batch()["input_ids"],
                           "attention_mask": batch()["attention_mask"]})
        endpoints = [StageEndpoint(i, LocalTransport(w.handle).call) for i, w in enumerate(workers)]
        recovered = PipelineTrainer(self.manifest, endpoints, trainer.checkpoints, restore_existing=True)
        recovered.resume()
        self.assertIsNone(workers[0].pending)
        self.assertEqual(recovered.train_batch(batch())["step"], 2)

    def test_cpu_checkpoint_can_initialize_offloaded_optimizer_masters(self):
        trainer, workers = self.trainer()
        trainer.train_batch(batch())
        saved = workers[0].snapshot()
        self.assertIsNone(saved["optimizer_master"])
        # Exercise the offload restore path without claiming a physical GPU test.
        target = StageWorker(self.manifest, 0)
        target.offloaded = True
        target.optimizer_parameters = [torch.nn.Parameter(p.detach().clone()) for p in target.trainable]
        target.optimizer = torch.optim.AdamW(target.optimizer_parameters, lr=0.001)
        target.scheduler = torch.optim.lr_scheduler.LambdaLR(target.optimizer, lambda _: 1.0)
        target.restore(saved)
        for master, parameter in zip(target.optimizer_parameters, target.trainable):
            torch.testing.assert_close(master, parameter, rtol=0, atol=0)
        self.assertEqual(target.step_number, 1)

    def test_remote_signed_http_pipeline_actual_update_and_resume(self):
        trainer_id, a, b = (Identity.generate(name) for name in ("trainer", "a", "b"))
        config = manifest()
        config["members"] = {i.member_id: i.public_key for i in (trainer_id, a, b)}
        workers = [StageWorker(config, i) for i in range(2)]
        servers = [RpcServer(w.handle, identity, config["members"], run_hash(config)).start()
                   for w, identity in zip(workers, (a, b))]
        try:
            client = RpcClient(trainer_id, config["members"], run_hash(config), timeout=10)
            endpoints = [StageEndpoint(i, lambda req, i=i: client.call(servers[i].url, req,
                                                                      expected_member=(a, b)[i].member_id))
                         for i in range(2)]
            trainer = PipelineTrainer(config, endpoints, CheckpointManager(self.root / "remote"))
            trainer.train_batch(batch())
            trainer.train_batch(batch())
            trainer.resume()
            self.assertEqual(trainer.train_batch(batch())["step"], 3)
        finally:
            for server in servers:
                server.close()


class AdmissionTests(unittest.TestCase):
    def test_no_active_parameter_memory_shortcut(self):
        self.assertFalse(place_stages([100, 100], [{"memory": {"available_bytes": 100}}])["feasible"])
        placed = place_stages([100, 100], [{"memory": {"available_bytes": 400}}])
        self.assertTrue(placed["feasible"])

    def test_run_hash_changes_for_dataset_train_mode_and_members(self):
        original = manifest()
        for key, new in (("mode", "full"), ("dataset", {"revision": "other"}), ("members", {"a": "public"})):
            changed = copy.deepcopy(original); changed[key] = new
            self.assertNotEqual(run_hash(original), run_hash(changed))
        original["model"]["config"]["use_cache"] = True
        with self.assertRaises(ValueError):
            validate_manifest(original)


if __name__ == "__main__":
    unittest.main()
