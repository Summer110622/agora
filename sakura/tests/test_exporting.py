import tempfile
import unittest
from pathlib import Path

import torch

from sakura.exporting import export_full_checkpoint
from sakura.model import build_model, build_stage, tiny_config


class FullExportTests(unittest.TestCase):
    def test_full_stage_shards_reload_as_exact_real_hf_text_model(self):
        torch.set_num_threads(1)
        config = tiny_config(vocab_size=64, hidden_size=32)
        expected = build_model(config, mode="full", seed=91)
        states = {str(i): {"mode": "full", "stage_index": i,
                           "model": build_stage(config, i, 2, mode="full", seed=91).state_dict()} for i in range(2)}
        with tempfile.TemporaryDirectory() as directory:
            target = export_full_checkpoint(states, config, 2, Path(directory) / "full")
            restored = build_model(config, mode="full", pretrained_path=target)
            for name, parameter in expected.named_parameters():
                torch.testing.assert_close(parameter, dict(restored.named_parameters())[name], rtol=0, atol=0)
            self.assertTrue((target / "model.safetensors.index.json").exists())


if __name__ == "__main__":
    unittest.main()
