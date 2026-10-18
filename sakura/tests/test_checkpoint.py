import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from sakura.auth import strict_json
from sakura.checkpoint import CheckpointError, CheckpointManager


RUN_HASH = "a" * 64


def stage(step, value):
    return {"step": step, "model": {"weight": torch.tensor([value], dtype=torch.float32)},
            "optimizer": {"state": {0: {"exp_avg": torch.zeros(1), "step": torch.tensor(step)}}},
            "scheduler": {"last_epoch": step}, "rng": torch.random.get_rng_state(),
            "run_hash": RUN_HASH, "stage_index": 0, "mode": "tiny"}


def trainer(step):
    return {"step": step, "cursor": {"record": step * 2, "token": 4},
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.random.get_rng_state()}}


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manager = CheckpointManager(self.root / "primary", mirrors=[self.root / "mirror"])

    def tearDown(self):
        self.temporary.cleanup()

    def commit(self, step=1):
        return self.manager.commit({"head": stage(step, 1), "tail": stage(step, 2)}, trainer(step),
                                   run_hash=RUN_HASH, step=step)

    def test_atomic_all_stage_roundtrip_rng_optimizer_cursor_and_mirror(self):
        manifest_path = self.commit()
        checkpoint = self.manager.load_current(expected_run_hash=RUN_HASH)
        self.assertEqual(set(checkpoint["stages"]), {"head", "tail"})
        self.assertTrue(torch.equal(checkpoint["stages"]["tail"]["model"]["weight"], torch.tensor([2.])))
        self.assertEqual(checkpoint["stages"]["head"]["optimizer"]["state"][0]["step"].item(), 1)
        self.assertEqual(checkpoint["training_state"]["cursor"]["record"], 2)
        self.assertIsInstance(checkpoint["training_state"]["rng"]["python"], tuple)
        self.assertTrue(np.array_equal(checkpoint["training_state"]["rng"]["numpy"][1], np.random.get_state()[1]))
        mirror = CheckpointManager(self.root / "mirror").load_current(RUN_HASH)
        self.assertEqual(mirror["manifest"], checkpoint["manifest"])
        self.assertTrue(any(path.suffix == ".safetensors" for path in manifest_path.parent.iterdir()))
        # Metadata stores tensor references rather than expanding model weights as base64.
        self.assertNotIn('"data":', (manifest_path.parent / "stage-000000.json").read_text())
        diagnostics = self.manager.mirror_diagnostics()
        self.assertTrue(diagnostics["distinct_directories"])
        self.assertFalse(diagnostics["physical_host_redundancy_verified"])

    def test_inconsistent_stages_and_wrong_run_rejected(self):
        with self.assertRaisesRegex(CheckpointError, "same checkpoint step"):
            self.manager.commit({"a": stage(1, 1), "b": stage(2, 2)}, trainer(1), run_hash=RUN_HASH, step=1)
        self.assertFalse((self.manager.directory / "CURRENT").exists())
        self.commit()
        with self.assertRaisesRegex(CheckpointError, "different run"):
            self.manager.load_current("b" * 64)

    def test_failed_write_never_advances_current(self):
        self.commit()
        previous = (self.manager.directory / "CURRENT").read_bytes()
        with mock.patch.object(self.manager, "_save_state", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.commit(2)
        self.assertEqual((self.manager.directory / "CURRENT").read_bytes(), previous)
        self.assertEqual(self.manager.load_current()["manifest"]["step"], 1)

    def test_failed_mirror_never_advances_primary_current(self):
        self.commit()
        previous = (self.manager.directory / "CURRENT").read_bytes()
        with mock.patch.object(self.manager, "_replicate", side_effect=OSError("mirror offline")):
            with self.assertRaisesRegex(CheckpointError, "Mirror replication failed"):
                self.commit(2)
        self.assertEqual((self.manager.directory / "CURRENT").read_bytes(), previous)
        self.assertEqual(self.manager.load_current()["manifest"]["step"], 1)
        self.assertTrue(self.manager.mirror_diagnostics()["errors"])

    def test_corrupted_shard_and_missing_stage_are_not_loadable(self):
        path = self.commit()
        manifest = strict_json(path.read_bytes())
        shard = path.parent / manifest["stages"]["head"]["tensors"]["file"]
        raw = bytearray(shard.read_bytes())
        raw[-1] ^= 1
        shard.write_bytes(raw)
        with self.assertRaisesRegex(CheckpointError, "corrupted"):
            self.manager.load_current()
        path = self.commit(2)
        manifest = strict_json(path.read_bytes())
        (path.parent / manifest["stages"]["tail"]["tensors"]["file"]).unlink()
        with self.assertRaises(CheckpointError):
            self.manager.load_current()

    def test_uncommitted_generation_not_selected_and_bad_manifest_rejected(self):
        path = self.commit()
        (self.manager.directory / "step-999999999999-orphan").mkdir()
        self.assertEqual(self.manager.load_current()["manifest"]["step"], 1)
        with path.open("ab") as stream:
            stream.write(b" ")
        with self.assertRaisesRegex(CheckpointError, "hash mismatch"):
            self.manager.load_current()

    def test_distinct_mirror_roots_enforced(self):
        with self.assertRaises(ValueError):
            CheckpointManager(self.root / "one", mirrors=[self.root / "one"])
        with self.assertRaises(ValueError):
            CheckpointManager(self.root / "one", mirrors=[self.root / "one" / "nested"])


if __name__ == "__main__":
    unittest.main()
