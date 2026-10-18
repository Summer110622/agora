import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from sakura.cli import main, build_parser
from sakura.manifest import load_manifest


class CliTests(unittest.TestCase):
    def call(self, *arguments):
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            result = main(list(arguments))
        self.assertEqual(result, 0, error.getvalue())
        return output.getvalue()

    def test_user_commands_init_train_restart_resume_export_infer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = json.loads(self.call("init", "--directory", str(root / "run")))
            self.assertTrue(first["fixture_only"])
            manifest = root / "run" / "run.json"
            self.call("train", "--manifest", str(manifest), "--steps", "3")
            self.call("resume", "--manifest", str(manifest), "--steps", "1")
            exported = json.loads(self.call("export", "--manifest", str(manifest), "--output", str(root / "adapter")))
            self.assertEqual(exported["step"], 4)
            self.assertTrue((root / "adapter" / "adapter_model.safetensors").exists())
            generated = json.loads(self.call("infer", "--manifest", str(manifest), "--prompt", "Add two numbers",
                                             "--max-new-tokens", "2"))
            self.assertLessEqual(len(generated["generated_token_ids"]), 2)
            self.assertEqual(generated["step"], 4)

    def test_experiment_cli_preserves_selected_topology_and_smoke_label(self):
        with tempfile.TemporaryDirectory() as directory:
            self.call("init", "--directory", directory, "--profile", "experiment", "--candidate", "3", "--mode", "full")
            manifest = load_manifest(Path(directory) / "run.json")
            self.assertEqual(manifest["model"]["config"]["num_experts"], 128)
            self.assertEqual(manifest["model"]["config"]["num_experts_per_tok"], 4)
            self.assertTrue(manifest["model"]["config"]["_sakura_experiment"]["synthetic_vocabulary"])

    def test_worker_default_remains_loopback(self):
        parsed = build_parser().parse_args(["join", "--manifest", "run.json", "--identity", "identity.private.json", "--stage", "0"])
        self.assertEqual(parsed.host, "127.0.0.1")
        self.assertFalse(parsed.allow_external)

    def test_raw_cpt_cli_updates_and_resumes_without_chat_formatting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = {"name": "hand-authored fixture", "url": "urn:sakura:cpt-fixture", "revision": "v1", "license": "CC0-1.0"}
            corpus = root / "documents.jsonl"
            corpus.write_text(json.dumps({"id": "raw", "language": "en", "repository": "sakura-fixtures",
                                          "problem_id": "raw", "text": "def add(a, b):\n    return a + b\n", "source": source,
                                          "fixture": True}) + "\n", encoding="utf-8")
            self.call("init", "--directory", str(root / "run"), "--objective", "cpt", "--mode", "full", "--data", str(corpus))
            manifest = root / "run" / "run.json"
            self.call("train", "--manifest", str(manifest), "--steps", "1")
            resumed = self.call("resume", "--manifest", str(manifest), "--steps", "1")
            self.assertIn('"cursor": 2', resumed)


if __name__ == "__main__":
    unittest.main()
