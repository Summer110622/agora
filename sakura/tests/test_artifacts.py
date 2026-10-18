import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from sakura.artifacts import (
    fingerprint_snapshot, fingerprint_tokenizer, verify_snapshot, verify_tokenizer,
)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "config.json").write_text('{"model_type":"qwen3_5_moe"}', encoding="utf-8")
        (self.root / "tokenizer.json").write_text('{"vocab":{"a":1}}', encoding="utf-8")
        (self.root / "tokenizer_config.json").write_text('{"pad_token":"<pad>"}', encoding="utf-8")
        (self.root / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
        (self.root / "model-00001.safetensors").write_bytes(b"small opaque weight fixture")
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.weight": "model-00001.safetensors"}}), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_content_map_is_portable_exact_and_deterministic(self):
        fingerprints = fingerprint_snapshot(self.root)
        self.assertEqual(list(fingerprints), sorted(fingerprints))
        self.assertTrue(all(not Path(name).is_absolute() for name in fingerprints))
        self.assertEqual(fingerprints, json.loads(json.dumps(fingerprints)))
        expected = (self.root / "model-00001.safetensors").read_bytes()
        self.assertEqual(fingerprints["model-00001.safetensors"],
                         {"sha256": hashlib.sha256(expected).hexdigest(), "bytes": len(expected)})
        verify_snapshot(self.root, fingerprints)
        (self.root / "README.md").write_text("not a model input", encoding="utf-8")
        self.assertEqual(fingerprints, fingerprint_snapshot(self.root))

    def test_digest_catches_same_length_weight_corruption(self):
        pinned = fingerprint_snapshot(self.root)
        file = self.root / "model-00001.safetensors"
        file.write_bytes(b"X" + file.read_bytes()[1:])
        with self.assertRaisesRegex(ValueError, "content mismatch"):
            verify_snapshot(self.root, pinned)

    def test_added_removed_artifacts_and_chat_template_changes_rejected(self):
        pinned = fingerprint_snapshot(self.root)
        extra = self.root / "extra.safetensors"
        extra.write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "list mismatch"):
            verify_snapshot(self.root, pinned)
        extra.unlink()
        template = self.root / "chat_template.jinja"
        template.write_text("{{ changed_messages }}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "content mismatch"):
            verify_snapshot(self.root, pinned)
        template.unlink()
        with self.assertRaisesRegex(ValueError, "list mismatch"):
            verify_snapshot(self.root, pinned)

    def test_index_and_expected_map_reject_traversal_and_missing_shards(self):
        index = self.root / "model.safetensors.index.json"
        for name in ("../outside.safetensors", "C:/outside.safetensors", "/outside.safetensors",
                     "nested\\outside.safetensors", "missing.safetensors"):
            with self.subTest(name=name):
                index.write_text(json.dumps({"weight_map": {"w": name}}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    fingerprint_snapshot(self.root)
        with self.assertRaises(ValueError):
            verify_snapshot(self.root, {"../outside": {"sha256": "0" * 64, "bytes": 0}})

    def test_expected_map_validates_sha_and_lengths_without_modification(self):
        expected = fingerprint_snapshot(self.root)
        original = copy.deepcopy(expected)
        verify_snapshot(self.root, expected)
        self.assertEqual(expected, original)
        for entry in ({"sha256": "claimed-upstream-revision", "bytes": 1},
                      {"sha256": "0" * 64, "bytes": True},
                      {"sha256": "0" * 64, "bytes": -1}):
            with self.assertRaises(ValueError):
                verify_snapshot(self.root, {"config.json": entry})

    def test_tokenizer_identity_does_not_require_or_hash_weights(self):
        artifacts = fingerprint_tokenizer(self.root)
        self.assertNotIn("model-00001.safetensors", artifacts)
        spec = {"kind": "huggingface", "revision": "a" * 40, "artifacts": artifacts}
        verify_tokenizer(spec, self.root)
        (self.root / "model-00001.safetensors").write_bytes(b"different weights")
        verify_tokenizer(spec, self.root)
        (self.root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "content mismatch"):
            verify_tokenizer(spec, self.root)
        with self.assertRaises(ValueError):
            verify_tokenizer({"kind": "huggingface", "revision": "a" * 40}, self.root)

    def test_scratch_revision_is_recomputed_and_file_hashes_verified(self):
        # Match experiments.train_tokenizer's identity construction without model
        # downloads or dependency-heavy tokenizer training in this focused test.
        (self.root / "config.json").unlink()
        hashes = {name: entry["sha256"] for name, entry in fingerprint_tokenizer(self.root).items()}
        metadata = {"schema_version": 1, "kind": "scratch_bytelevel_bpe", "artifact_sha256": hashes,
                    "corpus": [{"path": "machine-local-input"}], "output_directory": "machine-local-output",
                    "tokenizer_json": "machine-local-tokenizer", "actual_vocab_size": 262,
                    "smoke_test": True}
        identity = {name: value for name, value in metadata.items()
                    if name not in {"corpus", "output_directory", "tokenizer_json"}}
        revision = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                             ensure_ascii=False).encode()).hexdigest()
        metadata["revision"] = revision
        file = self.root / "sakura-tokenizer.json"
        file.write_text(json.dumps(metadata), encoding="utf-8")
        spec = {"kind": "huggingface", "revision": revision}
        verify_tokenizer(spec, self.root)
        metadata["output_directory"] = "a different local path"
        file.write_text(json.dumps(metadata), encoding="utf-8")
        verify_tokenizer(spec, self.root)
        metadata["actual_vocab_size"] = 999
        file.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "revision"):
            verify_tokenizer(spec, self.root)
        metadata["actual_vocab_size"] = 262
        file.write_text(json.dumps(metadata), encoding="utf-8")
        (self.root / "tokenizer.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "content mismatch"):
            verify_tokenizer(spec, self.root)

    def test_symlink_rejected_when_host_allows_creation(self):
        link = self.root / "linked-tokenizer.json"
        try:
            link.symlink_to(self.root / "tokenizer.json")
        except (OSError, NotImplementedError):
            self.skipTest("Host does not grant symlink creation")
        with self.assertRaisesRegex(ValueError, "symlink"):
            fingerprint_snapshot(self.root)

    def test_byte_fixture_has_explicit_identity(self):
        verify_tokenizer({"kind": "byte", "revision": "fixture-utf8-v1"}, None)
        with self.assertRaises(ValueError):
            verify_tokenizer({"kind": "byte", "revision": "unknown"}, None)


if __name__ == "__main__":
    unittest.main()
