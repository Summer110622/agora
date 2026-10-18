import ast
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from sakura.data import (IGNORE_INDEX, SFTRecord, content_digest, dataset_manifest,
                         encode_record, encode_windows, load_records, prepare_dataset)
from sakura.evaluation import (SandboxLimits, bits_per_byte, evaluate_python,
                               summarize_evaluation)


def record(identifier="r1", repo="fixture/repo1", problem="fixture:p1", language="en", answer="return a + b"):
    return SFTRecord.from_dict({"id": identifier, "language": language, "repository": repo,
                                "problem_id": problem,
                                "messages": [{"role": "user", "content": "Add two numbers."},
                                             {"role": "assistant", "content": answer}],
                                "source": {"name": "test fixture", "url": "urn:sakura:fixture:" + identifier,
                                           "revision": "test-v1", "license": "CC0-1.0"}, "fixture": True})


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 3
    chat_template = "original fixture template"

    def __init__(self, native_mask=True):
        self.native_mask = native_mask

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt=False,
                            return_dict=False, return_assistant_tokens_mask=False):
        text, mask = "", []
        for message in messages:
            header = "<" + message["role"] + ">\n"
            suffix = message["content"] + "\n</" + message["role"] + ">\n"
            text += header + suffix
            mask += [0] * len(header) + [int(message["role"] == "assistant")] * len(suffix)
        if add_generation_prompt:
            text += "<assistant>\n"
            mask += [0] * len("<assistant>\n")
        ids = [ord(char) for char in text]
        if return_dict:
            return {"input_ids": ids, "assistant_masks": mask if self.native_mask else [0] * len(mask)}
        return ids if tokenize else text


class DataTests(unittest.TestCase):
    def test_provenance_is_required(self):
        raw = record().to_dict()
        raw["messages"] = [vars(message) for message in record().messages]
        raw["source"]["license"] = ""
        with self.assertRaisesRegex(ValueError, "license"):
            SFTRecord.from_dict(raw)

    def test_schema_roundtrips_and_revision_covers_removed_duplicate_metadata(self):
        from dataclasses import replace
        original = record("a")
        self.assertEqual(SFTRecord.from_dict(original.to_dict()), original)
        duplicate = replace(original, id="b", repository="fixture/duplicate-repo")
        records = [original, duplicate]
        first = dataset_manifest(records, prepare_dataset(records, train_fraction=1, validation_fraction=0))
        changed = [original, replace(duplicate, id="c")]
        second = dataset_manifest(changed, prepare_dataset(changed, train_fraction=1, validation_fraction=0))
        self.assertNotEqual(first["data_revision"], second["data_revision"])

    def test_load_reports_line_and_rejects_repeated_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text(json.dumps(record().to_dict()) + "\n{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r":2:"):
                load_records(path)
            path.write_text((json.dumps(record().to_dict()) + "\n") * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                load_records(path)

    def test_split_dedup_connects_removed_duplicate_repositories(self):
        records = [record("a", "repo/a", "p/a"),
                   record("b", "repo/b", "p/b"),  # duplicate text connects repo/a and repo/b
                   record("c", "repo/b", "p/c", answer="return a - b"),
                   record("d", "repo/d", "p/c", language="ja", answer="return a * b")]
        splits = prepare_dataset(records)
        retained = [(split, item) for split, values in splits.items() for item in values]
        self.assertEqual(len(retained), 3)
        self.assertEqual(len({split for split, item in retained}), 1)
        combined = next(item for _, item in retained if item.id == "a")
        self.assertEqual(combined.additional_sources, (records[1].source,))
        manifest = dataset_manifest(records, splits)
        self.assertEqual(manifest["duplicates_removed"], 1)
        self.assertTrue(manifest["contains_fixtures"])
        self.assertEqual(len(manifest["sources"]), 4)
        self.assertTrue(all(value == 0 for value in manifest["content_overlap"].values()))

    def test_split_stable_under_reordering_and_group_safe(self):
        records = [record(str(index), f"repo/{index // 2}", f"p/{index // 2}",
                          language="ja" if index % 2 else "en", answer=f"return {index}") for index in range(100)]
        splits = prepare_dataset(records)
        self.assertEqual(splits, prepare_dataset(reversed(records)))
        self.assertEqual(sum(map(len, splits.values())), 100)
        self.assertEqual(set(splits), {"train", "validation", "test"})
        self.assertTrue(all(splits.values()))
        first_revision = dataset_manifest(records, splits)["data_revision"]
        self.assertEqual(first_revision, dataset_manifest(reversed(records), prepare_dataset(reversed(records)))["data_revision"])
        seen = {}
        for split, values in splits.items():
            for item in values:
                self.assertEqual(seen.setdefault(item.repository, split), split)

    def test_contamination_audit_rejects_external_bad_splits(self):
        a, b = record("a"), record("b", answer="return a - b")
        with self.assertRaisesRegex(ValueError, "group contamination"):
            dataset_manifest([a, b], {"train": [a], "test": [b]})
        with self.assertRaisesRegex(ValueError, "content contamination"):
            dataset_manifest([a], {"train": [a], "test": [a]})

    def test_code_indentation_not_destroyed_by_dedup(self):
        self.assertNotEqual(content_digest(record(answer="def f():\n    return 1")),
                            content_digest(record(answer="def f():\n return 1")))
        self.assertEqual(content_digest(record(answer="return 1\r\n")), content_digest(record(answer="return 1\n")))

    def test_assistant_labels_and_padding_original_tokenizer_unchanged(self):
        tokenizer = FakeTokenizer()
        before = copy.deepcopy(vars(tokenizer))
        encoded = encode_record(record(), tokenizer, 128)
        self.assertEqual(len(encoded["input_ids"]), 128)
        prefix_length = len("<user>\nAdd two numbers.\n</user>\n<assistant>\n")
        self.assertTrue(all(label == IGNORE_INDEX for label in encoded["labels"][:prefix_length]))
        self.assertEqual(encoded["labels"][prefix_length], ord("r"))
        self.assertTrue(all(label == IGNORE_INDEX for label, present in zip(encoded["labels"], encoded["attention_mask"]) if not present))
        self.assertEqual(vars(tokenizer), before)

    def test_no_generation_tag_fallback_matches_native_mask(self):
        self.assertEqual(encode_record(record(), FakeTokenizer(native_mask=False), 128),
                         encode_record(record(), FakeTokenizer(native_mask=True), 128))

    def test_windows_keep_every_assistant_target_once(self):
        sample = record(answer="def f(a, b):\n    return a + b\n" * 10)
        tokenizer = FakeTokenizer()
        native = tokenizer.apply_chat_template([vars(m) for m in sample.messages], tokenize=True,
                                              return_dict=True, return_assistant_tokens_mask=True)
        windows = encode_windows(sample, tokenizer, 32)
        self.assertGreater(len(windows), 1)
        targets = [label for window in windows for label in window["labels"] if label != IGNORE_INDEX]
        expected = [token for token, assistant in zip(native["input_ids"], native["assistant_masks"]) if assistant]
        self.assertEqual(targets, expected)
        self.assertTrue(all(window["labels"][0] == IGNORE_INDEX for window in windows))
        with self.assertRaisesRegex(ValueError, "multiple windows"):
            encode_record(sample, tokenizer, 32)


class EvaluationTests(unittest.TestCase):
    def test_bits_per_byte_uses_summed_loss_and_raw_utf8_bytes(self):
        self.assertAlmostEqual(bits_per_byte(10 * math.log(2), len("日本語".encode("utf-8"))), 10 / 9)
        for args in ((-1, 3), (math.nan, 3), (1, 0)):
            with self.assertRaises(ValueError):
                bits_per_byte(*args)

    def test_bilingual_pairs_and_matched_regressions(self):
        results = [{"problem_id": "p1", "language": "ja", "executed": True, "passed": False,
                    "nll_nats": math.log(2) * 6, "utf8_bytes": 3},
                   {"problem_id": "p1", "language": "en", "executed": True, "passed": True,
                    "nll_nats": math.log(2) * 2, "utf8_bytes": 1},
                   {"problem_id": "p2", "language": "en", "executed": False, "passed": None}]
        baseline = [{"problem_id": "p1", "language": lang, "executed": True, "passed": True} for lang in ("ja", "en")]
        report = summarize_evaluation(results, baseline)
        self.assertAlmostEqual(report["bits_per_byte"], 2)
        self.assertEqual(report["paired"]["problems"], 1)
        self.assertEqual(report["paired"]["balanced_pass_at_1"], 0.5)
        self.assertEqual(report["comparison"]["regressions"], [{"problem_id": "p1", "language": "ja"}])
        self.assertEqual(report["per_language"]["en"]["executed"], 1)
        self.assertEqual(report["per_language"]["en"]["pass_at_1"], 1)

    def test_unexecuted_cannot_claim_pass_and_duplicate_samples_rejected(self):
        with self.assertRaisesRegex(ValueError, "unexecuted"):
            summarize_evaluation([{"problem_id": "p1", "language": "en", "passed": True}])
        sample = {"problem_id": "p1", "language": "en", "executed": True, "passed": True}
        with self.assertRaisesRegex(ValueError, "exactly one"):
            summarize_evaluation([sample, sample])

    def test_execution_is_skipped_by_default_and_no_host_process_runs(self):
        with mock.patch("sakura.evaluation.subprocess.Popen") as process:
            value = evaluate_python("raise RuntimeError()", "assert False")
        self.assertFalse(value["executed"])
        self.assertIsNone(value["passed"])
        process.assert_not_called()

    def test_sandbox_requires_digest_local_docker_and_never_falls_back(self):
        with self.assertRaisesRegex(ValueError, "digest"):
            evaluate_python("", "assert True", execute=True, limits=SandboxLimits(image="python:latest"))
        with mock.patch("sakura.evaluation.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "host execution is disabled"):
                evaluate_python("", "assert True", execute=True, limits=SandboxLimits(image="python@sha256:" + "a" * 64))

    def test_docker_flags_bounded_runner_and_cleanup(self):
        image = "python@sha256:" + "a" * 64
        captured = {}

        def run(command, payload, limits):
            captured["command"] = command
            marker = ast.literal_eval(payload.decode().splitlines()[-1][6:-1])
            return 0, marker + "\n", "", "completed"

        success = subprocess_result = mock.Mock(returncode=0, stderr=b"")
        with mock.patch("sakura.evaluation.shutil.which", return_value="docker"), \
                mock.patch("sakura.evaluation.subprocess.run", return_value=success) as invocations, \
                mock.patch("sakura.evaluation._bounded_process", side_effect=run):
            result = evaluate_python("def f(): return 1", "assert f() == 1", execute=True, limits=SandboxLimits(image=image))
        self.assertTrue(result["passed"])
        command = captured["command"]
        for flag in ("--network=none", "--read-only", "--pull=never", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                     "--memory=256m", "--memory-swap=256m", "--pids-limit=64"):
            self.assertIn(flag, command)
        self.assertNotIn("--mount", command)
        self.assertNotIn("--privileged", command)
        self.assertIn("--force", invocations.call_args_list[-1].args[0])

    def test_early_exit_without_harness_completion_is_failure(self):
        with mock.patch("sakura.evaluation.shutil.which", return_value="docker"), \
                mock.patch("sakura.evaluation.subprocess.run", return_value=mock.Mock(returncode=0, stderr=b"")), \
                mock.patch("sakura.evaluation._bounded_process", return_value=(0, "", "", "completed")):
            result = evaluate_python("raise SystemExit(0)", "assert False", execute=True,
                                     limits=SandboxLimits(image="python@sha256:" + "a" * 64))
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
