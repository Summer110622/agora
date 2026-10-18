from itertools import islice
import json
import math
from pathlib import Path
import tempfile
import unittest

from sakura.corpus import encode_cpt_windows, iter_training_windows, load_cpt_records
from sakura.data import (Message, SFTRecord, SourceProvenance, dataset_manifest,
                         encode_windows, prepare_dataset)


SOURCE = {"name": "hand-authored raw-text unit fixture", "url": "urn:sakura:cpt-unit-fixture",
          "revision": "fixture-v1", "license": "CC0-1.0"}


def raw(identifier="a", text="def add(a, b):\n    return a + b\n", repo="fixture/repo", problem="fixture:add"):
    return {"id": identifier, "language": "en", "repository": repo, "problem_id": problem,
            "text": text, "source": SOURCE, "fixture": True}


def audit_record(identifier, text):
    return SFTRecord(identifier, "en", "fixture/" + identifier, "fixture:" + identifier,
                     (Message("assistant", text),), SourceProvenance.from_dict(SOURCE), True)


class RawByteTokenizer:
    pad_token_id, eos_token_id = 0, 2

    def __init__(self):
        self.encoded_texts = []

    def encode(self, text, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("CPT must not add automatic special tokens")
        self.encoded_texts.append(text)
        return [value + 3 for value in text.encode("utf-8")]

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("CPT must never call a chat template")


class CorpusTests(unittest.TestCase):
    def test_raw_text_roundtrip_provenance_and_shared_split_audit(self):
        text = "# 日本語の例\r\ndef add(a, b):\n    return a + b\n\n"
        rows = [raw("a", text), raw("b", text, repo="fixture/duplicate", problem="fixture:duplicate"),
                raw("c", "print('different')", repo="fixture/duplicate", problem="fixture:other")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
            records = load_cpt_records(path)
        self.assertEqual(records[0].messages, (Message("assistant", text),))
        self.assertEqual(records[0].source.license, "CC0-1.0")
        self.assertTrue(records[0].fixture)
        splits = prepare_dataset(records)
        self.assertEqual(sum(map(len, splits.values())), 2)
        self.assertEqual(sum(bool(items) for items in splits.values()), 1)
        audit = dataset_manifest(records, splits)
        self.assertEqual(audit["duplicates_removed"], 1)
        self.assertTrue(audit["contains_fixtures"])
        self.assertTrue(all(count == 0 for count in audit["content_overlap"].values()))

    def test_load_rejects_missing_provenance_empty_text_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            for invalid in ({**raw(), "source": {}}, {**raw(), "text": "  \n"}, {**raw(), "fixture": "yes"}):
                path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, ":1:"):
                    load_cpt_records(path)
            path.write_text((json.dumps(raw()) + "\n") * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                load_cpt_records(path)

    def test_raw_windows_eos_context_padding_and_every_target_once(self):
        text = "日本語の足し算\n    return a + b\n"
        record, tokenizer = audit_record("a", text), RawByteTokenizer()
        windows = list(encode_cpt_windows(record, tokenizer, 11))
        self.assertGreater(len(windows), 1)
        ids = [byte + 3 for byte in text.encode("utf-8")] + [tokenizer.eos_token_id]
        actual_targets = [label for window in windows for label in window["labels"] if label != -100]
        self.assertEqual(actual_targets, ids[1:])
        self.assertEqual(actual_targets[-1], tokenizer.eos_token_id)
        self.assertEqual(tokenizer.encoded_texts, [text])
        self.assertTrue(all(window["labels"][0] == -100 for window in windows))
        for previous, following in zip(windows, windows[1:]):
            self.assertEqual(previous["input_ids"][-1], following["input_ids"][0])
        for window in windows:
            self.assertEqual(len(window["input_ids"]), 11)
            self.assertTrue(all(label == -100 for label, mask in zip(window["labels"], window["attention_mask"]) if mask == 0))

    def test_cpt_labels_work_with_shifted_causal_loss(self):
        import torch
        from sakura.model import causal_lm_loss
        batch = next(encode_cpt_windows(audit_record("a", "abc"), RawByteTokenizer(), 8))
        logits = torch.zeros(1, 8, 260, requires_grad=True)
        labels = torch.tensor([batch["labels"]])
        loss = causal_lm_loss(logits, labels)
        self.assertAlmostEqual(float(loss.detach()), math.log(260), places=5)
        loss.backward()
        self.assertGreater(float(logits.grad[:, :3].abs().sum()), 0)
        self.assertEqual(float(logits.grad[:, 3:].abs().sum()), 0)

    def test_training_windows_do_not_pack_or_tokenize_entire_corpus_upfront(self):
        records = [audit_record("a", "abcdefghij"), audit_record("b", "second document")]
        tokenizer = RawByteTokenizer()
        stream = iter_training_windows(records, tokenizer, 5, objective="cpt")
        first = next(stream)
        self.assertEqual(tokenizer.encoded_texts, ["abcdefghij"])
        second = next(stream)
        self.assertEqual(tokenizer.encoded_texts, ["abcdefghij"])
        self.assertEqual(first["input_ids"][-1], second["input_ids"][0])
        # A record has three independent windows; the fourth begins document b.
        next(stream)
        fourth = next(stream)
        self.assertEqual(tokenizer.encoded_texts, ["abcdefghij", "second document"])
        self.assertEqual(fourth["input_ids"][0], ord("s") + 3)

    def test_resuming_global_cursor_matches_deterministic_epoch_stream(self):
        records = [audit_record("a", "abcdefghij"), audit_record("b", "second document")]
        baseline = list(islice(iter_training_windows(records, RawByteTokenizer(), 5, objective="cpt"), 35))
        resumed = list(islice(iter_training_windows(records, RawByteTokenizer(), 5, objective="cpt", start_cursor=19), 16))
        self.assertEqual(resumed, baseline[19:])
        # Large cursors skip at most a complete epoch, not millions of epochs.
        tokenizer = RawByteTokenizer()
        actual = next(iter_training_windows(records, tokenizer, 5, objective="cpt", start_cursor=1_000_000))
        per_epoch = sum(len(list(encode_cpt_windows(record, RawByteTokenizer(), 5))) for record in records)
        self.assertEqual(actual, baseline[1_000_000 % per_epoch])
        self.assertLessEqual(len(tokenizer.encoded_texts), 4)

    def test_sft_stream_matches_existing_masks_and_windows(self):
        from sakura.tokenization import FixtureByteTokenizer
        record = SFTRecord.from_dict({"id": "sft", "language": "ja", "repository": "fixture/repo",
                                     "problem_id": "fixture:p", "source": SOURCE, "fixture": True,
                                     "messages": [{"role": "user", "content": "Pythonの足し算を書いて。"},
                                                  {"role": "assistant", "content": "def add(a, b):\n    return a + b"}]})
        tokenizer = FixtureByteTokenizer()
        expected = encode_windows(record, tokenizer, 16)
        stream = iter_training_windows([record], tokenizer, 16)
        self.assertEqual(list(islice(stream, len(expected) * 2)), expected + expected)

    def test_invalid_objective_empty_corpus_and_wrong_record_kind_fail(self):
        for kwargs in ({"objective": "unknown"}, {"start_cursor": -1}, {"seq": 1}):
            options = {"seq": 8, "objective": "cpt", **kwargs}
            with self.assertRaises(ValueError):
                next(iter_training_windows([audit_record("a", "text")], RawByteTokenizer(), **options))
        with self.assertRaisesRegex(ValueError, "no records"):
            next(iter_training_windows([], RawByteTokenizer(), 8, objective="cpt"))
        with self.assertRaisesRegex(ValueError, "objective=sft"):
            next(iter_training_windows([audit_record("a", "text")], RawByteTokenizer(), 8, objective="sft"))
        tokenizer = RawByteTokenizer()
        tokenizer.eos_token_id = None
        with self.assertRaisesRegex(ValueError, "eos_token_id"):
            next(encode_cpt_windows(audit_record("a", "text"), tokenizer, 8))


if __name__ == "__main__":
    unittest.main()
