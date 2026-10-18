import math
from pathlib import Path
import tempfile
import unittest

from sakura.experiments import (candidate_to_model_config, estimate_capacity, experiment_grid,
                                project_schedule, select_experiment, train_tokenizer)


def result(identifier="full-e64-k2-v64000", ja=0.5, en=0.5, bpb=1.5, speed=10):
    return {"experiment_id": identifier, "evaluation_revision": "fixed-paired-fixture-v1",
            "compute_budget_flops": 1000000, "paired_executed_problems": 10,
            "japanese_pass_at_1": ja, "english_pass_at_1": en,
            "validation_bits_per_byte": bpb, "throughput_tokens_per_second": speed}


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_grid_has_all_12_and_equal_expert_capacity(self):
        grid = experiment_grid()
        self.assertEqual(len(grid), 12)
        self.assertEqual(len({c["experiment_id"] for c in grid}), 12)
        self.assertEqual({c["attention"] for c in grid}, {"full", "hybrid"})
        self.assertEqual({(c["experts"], c["top_k"]) for c in grid}, {(64, 2), (128, 4), (256, 8)})
        self.assertEqual({c["vocab_size"] for c in grid}, {64000, 128000})
        self.assertEqual(len({c["capacity"]["expert_parameters"] for c in grid}), 1)
        self.assertEqual(len({c["capacity"]["active_expert_parameters"] for c in grid}), 1)
        self.assertTrue(all(c["capacity"]["estimated"] for c in grid))

    def test_capacity_includes_entire_weight_table_and_explicit_linear_cost(self):
        config = {"hidden_size": 16, "layers": 4, "experts": 64, "top_k": 2,
                  "expert_width": 8, "vocab_size": 64, "attention": "hybrid",
                  "linear_attention_parameters_per_layer": 512}
        capacity = estimate_capacity(config)
        shared = 64 * 16 + 1 * 4 * 16 * 16 + 3 * 512 + 4 * (16 * 64 + 2 * 16) + 16
        self.assertEqual(capacity["total_parameters"], shared + 4 * 64 * 3 * 16 * 8)
        self.assertEqual(capacity["active_parameters"], shared + 4 * 2 * 3 * 16 * 8)
        self.assertEqual(capacity["weight_bytes_bf16"], 2 * capacity["total_parameters"])
        self.assertGreater(capacity["total_parameters"], capacity["active_parameters"])
        self.assertFalse(capacity["attention_cost_is_proxy"])

    def test_grid_rejects_rounded_unequal_experts(self):
        with self.assertRaisesRegex(ValueError, "multiple of four"):
            experiment_grid(base_expert_width=127)

    def test_candidate_maps_to_actual_hybrid_or_full_hf_config(self):
        grid = experiment_grid(base_expert_width=8)
        for candidate in grid:
            config = candidate_to_model_config(candidate, synthetic_vocab_size=512)
            self.assertEqual(config["hidden_size"], 64)
            self.assertEqual(config["num_hidden_layers"], 4)
            self.assertEqual(config["num_experts"], candidate["experts"])
            self.assertEqual(config["num_experts_per_tok"], candidate["top_k"])
            self.assertEqual(config["moe_intermediate_size"], candidate["expert_width"])
            self.assertEqual(config["vocab_size"], 512)
            self.assertEqual(candidate["vocab_size"], config["_sakura_experiment"]["candidate_vocab_size"])
            self.assertTrue(config["_sakura_experiment"]["synthetic_vocabulary"])
            self.assertFalse(config["tie_word_embeddings"])
            expected = ["full_attention"] * 4 if candidate["attention"] == "full" else ["linear_attention"] * 3 + ["full_attention"]
            self.assertEqual(config["layer_types"], expected)
        normal = candidate_to_model_config(grid[0], proxy=False)
        self.assertEqual(normal["hidden_size"], grid[0]["hidden_size"])
        self.assertEqual(normal["vocab_size"], grid[0]["vocab_size"])
        self.assertFalse(normal["_sakura_experiment"]["synthetic_vocabulary"])

    def test_actual_candidate_full_and_hybrid_can_update_weights(self):
        import torch
        from sakura.model import build_model

        for attention in ("full", "hybrid"):
            candidate = next(c for c in experiment_grid(base_expert_width=8) if c["attention"] == attention)
            config = candidate_to_model_config(candidate, synthetic_vocab_size=512)
            model = build_model(config, mode="full", seed=7)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            ids = torch.tensor([[20, 21, 22, 23, 24, 25, 26, 27]])
            before = model.lm_head.weight.detach().clone()
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids, use_cache=False)
            self.assertTrue(torch.isfinite(output.loss))
            output.loss.backward()
            optimizer.step()
            self.assertFalse(torch.equal(model.lm_head.weight.detach(), before))
            self.assertEqual(output.logits.shape, (1, 8, 512))

    def test_selection_priority_and_deterministic_ties(self):
        # Quality dominates validation loss and speed; lower BPB breaks quality ties.
        values = [result("fast", ja=0.4, en=0.4, bpb=0.1, speed=1000),
                  result("good", ja=0.7, en=0.5, bpb=1.8, speed=50),
                  result("best", ja=0.5, en=0.7, bpb=1.7, speed=1)]
        self.assertEqual(select_experiment(values)["experiment_id"], "best")
        values = [result("b", speed=20), result("a", speed=20), result("c", speed=10)]
        self.assertEqual(select_experiment(reversed(values))["experiment_id"], "a")

    def test_selection_requires_real_paired_evidence_and_comparable_budgets(self):
        for key, invalid in (("paired_executed_problems", 0), ("japanese_pass_at_1", None),
                             ("validation_bits_per_byte", math.nan)):
            candidate = result()
            candidate[key] = invalid
            with self.assertRaises(ValueError):
                select_experiment([candidate])
        other = result("other")
        other["compute_budget_flops"] *= 2
        with self.assertRaisesRegex(ValueError, "same fixed"):
            select_experiment([result(), other])

    def test_schedule_uses_full_measured_week_zero_downtime_and_30_percent_margin(self):
        observations = [{"started_at": "2026-01-01T00:00:00Z", "ended_at": "2026-01-04T00:00:00Z", "tokens": 0},
                        {"started_at": "2026-01-04T00:00:00Z", "ended_at": "2026-01-08T00:00:00Z", "tokens": 6048000}]
        schedule = project_schedule(observations, target_tokens=7000000, completed_tokens=952000)
        self.assertEqual(schedule["observed_days"], 7)
        self.assertEqual(schedule["observed_tokens_per_second"], 10)
        self.assertEqual(schedule["effective_tokens_per_second"], 7)
        self.assertEqual(schedule["projected_days"], 10)
        self.assertNotIn("completion_date", schedule)

    def test_schedule_rejects_short_fake_gapped_overlapping_or_naive_evidence(self):
        invalid_sets = [[],
                        [{"started_at": "2026-01-01T00:00:00Z", "ended_at": "2026-01-07T00:00:00Z", "tokens": 1}],
                        [{"started_at": "2026-01-01T00:00:00", "ended_at": "2026-01-08T00:00:00", "tokens": 1}],
                        [{"started_at": "2026-01-01T00:00:00Z", "ended_at": "2026-01-08T00:00:00Z", "tokens": 1},
                         {"started_at": "2026-01-09T00:00:00Z", "ended_at": "2026-01-10T00:00:00Z", "tokens": 1}],
                        [{"started_at": "2026-01-01T00:00:00Z", "ended_at": "2026-01-08T00:00:00Z", "tokens": 1},
                         {"started_at": "2026-01-07T00:00:00Z", "ended_at": "2026-01-10T00:00:00Z", "tokens": 1}]]
        for observations in invalid_sets:
            with self.assertRaises(ValueError):
                project_schedule(observations, target_tokens=100)


class TokenizerTests(unittest.TestCase):
    provenance = [{"name": "handwritten unit fixture, not a training corpus", "url": "urn:sakura:tokenizer-fixture",
                   "revision": "fixture-v1", "license": "CC0-1.0"}]

    def test_actual_bytelevel_training_hashes_roundtrip_and_assistant_mask(self):
        from transformers import AutoTokenizer
        from sakura.data import SFTRecord, encode_record

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "train.txt"
            text = "def add(a, b):\n    return a + b\n日本語の足し算の説明です。\n"
            corpus.write_text(text * 8, encoding="utf-8")
            before = corpus.read_bytes()
            metadata = train_tokenizer([corpus], root / "scratch-tokenizer", provenance=self.provenance,
                                       vocab_size=300, smoke_test=True, min_frequency=1)
            self.assertEqual(metadata["actual_vocab_size"], 300)
            self.assertTrue(metadata["requested_vocabulary_reached"])
            self.assertEqual(corpus.read_bytes(), before)
            self.assertEqual(len(metadata["corpus_sha256"]), 64)
            self.assertEqual(len(metadata["training_iterator_sha256"]), 64)
            self.assertEqual(metadata["counts"]["utf8_bytes"], len(before))
            tokenizer = AutoTokenizer.from_pretrained(str(root / "scratch-tokenizer"), local_files_only=True)
            ids = tokenizer.encode(text, add_special_tokens=False)
            self.assertEqual(tokenizer.decode(ids, skip_special_tokens=False), text)
            sample = SFTRecord.from_dict({"id": "fixture", "language": "ja", "repository": "fixture/repo",
                                        "problem_id": "fixture:add", "fixture": True, "source": self.provenance[0],
                                        "messages": [{"role": "user", "content": "足し算を書いて。"},
                                                     {"role": "assistant", "content": "def add(a, b):\n    return a + b"}]})
            batch = encode_record(sample, tokenizer, 128)
            self.assertTrue(any(label != -100 for label in batch["labels"]))
            self.assertTrue(all(label == -100 for label, mask in zip(batch["labels"], batch["attention_mask"]) if not mask))
            with self.assertRaises(FileExistsError):
                train_tokenizer([corpus], root / "scratch-tokenizer", provenance=self.provenance,
                                vocab_size=300, smoke_test=True)
            # Same frozen data and parameters produce the same tokenizer revision.
            copy = train_tokenizer([corpus], root / "scratch-tokenizer-copy", provenance=self.provenance,
                                   vocab_size=300, smoke_test=True, min_frequency=1)
            self.assertEqual(metadata["revision"], copy["revision"])

    def test_vocab_targets_provenance_and_heldout_guard(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = root / "data.txt"
            text.write_text("fixture text", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "smoke_test=True"):
                train_tokenizer([text], root / "small", provenance=self.provenance, vocab_size=300)
            with self.assertRaisesRegex(ValueError, "provenance"):
                train_tokenizer([text], root / "unknown-source", provenance=[], vocab_size=300, smoke_test=True)
            with self.assertRaisesRegex(ValueError, "below requested"):
                train_tokenizer([text], root / "insufficient", provenance=self.provenance)
            self.assertFalse((root / "insufficient").exists())
            raw = {"id": "heldout", "language": "en", "repository": "fixture/repo", "problem_id": "fixture:p",
                   "source": self.provenance[0], "split": "test",
                   "messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]}
            path = root / "heldout.jsonl"
            path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(Exception, "validation or test"):
                train_tokenizer([path], root / "leaked", provenance=self.provenance, input_format="sft_jsonl",
                                vocab_size=300, smoke_test=True)
            self.assertFalse((root / "leaked").exists())


if __name__ == "__main__":
    unittest.main()
