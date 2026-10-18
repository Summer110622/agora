import unittest

import torch

from sakura.capacity import admit_run, measure_model_capacity
from sakura.model import build_model, tiny_config


def fixture_config(**overrides):
    return tiny_config(vocab_size=67, hidden_size=16, num_hidden_layers=4,
                       num_attention_heads=2, num_key_value_heads=1,
                       linear_num_key_heads=1, linear_num_value_heads=2,
                       moe_intermediate_size=8, shared_expert_intermediate_size=8,
                       num_experts=4, lora={"rank": 4, "alpha": 8}, **overrides)


def doctor(host_id, ram=10**12, disk=10**12, backend="cpu", vram=None):
    value = {"host_id": host_id, "memory": {"available_bytes": ram}, "disk": {"free_bytes": disk},
             "backend": backend, "device": backend}
    if vram is not None:
        value["accelerator"] = {"free_bytes": vram}
    return value


class CapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_meta_counts_equal_actual_full_and_lora_models_without_weight_storage(self):
        for mode in ("full", "lora"):
            config = fixture_config()
            actual = build_model(config, mode=mode, seed=5)
            report = measure_model_capacity(config, num_stages=2, mode=mode, sequence_length=8)
            self.assertEqual(report["total_parameters"], sum(p.numel() for p in actual.parameters()))
            self.assertEqual(report["trainable_parameters"], sum(p.numel() for p in actual.parameters() if p.requires_grad))
            self.assertEqual(report["materialized_model_weight_bytes"], 0)
            self.assertTrue(report["parameter_counts_exact_for_config"])
            self.assertEqual(sum(s["parameters"] for s in report["stages"]), report["total_parameters"])
            self.assertTrue(all(s["parameter_storage_device"] == "meta" for s in report["stages"]))
            self.assertEqual(report["cpu_model_bytes_fp32"], 4 * report["total_parameters"])
            self.assertFalse(report["training_peak_measured"])

    def test_active_fraction_does_not_reduce_full_expert_weight_storage(self):
        config = fixture_config()
        config["num_experts_per_tok"] = 1
        sparse = measure_model_capacity(config, 2, mode="full")
        config["num_experts_per_tok"] = 4
        dense = measure_model_capacity(config, 2, mode="full")
        self.assertLess(sparse["structural_active_parameters"], dense["structural_active_parameters"])
        self.assertEqual(sparse["total_parameters"], dense["total_parameters"])
        self.assertEqual(sparse["cpu_model_bytes_fp32"], dense["cpu_model_bytes_fp32"])
        self.assertEqual(dense["structural_active_parameters"], dense["total_parameters"])
        self.assertEqual(sparse["cpu_model_bytes_fp32"], 4 * sparse["total_parameters"])

    def test_cpu_adam_full_training_and_checkpoint_gather_are_accounted(self):
        report = measure_model_capacity(fixture_config(), 2, mode="full", sequence_length=8)
        for stage in report["stages"]:
            count = stage["parameters"]
            self.assertEqual(stage["optimizer_state_bytes"], 8 * count)
            self.assertEqual(stage["optimizer_master_bytes"], 4 * count)
            self.assertEqual(stage["gradients_bytes"], 4 * count)
            self.assertEqual(stage["cpu_training_lower_bound_bytes"], 16 * count + stage["activation_lower_bound_bytes"])
            self.assertGreaterEqual(stage["minimum_host_bytes"], 16 * count + stage["worker_checkpoint_copy_bytes_cpu"])
        self.assertEqual(report["coordinator_checkpoint_gather_lower_bound_bytes"],
                         sum(stage["worker_checkpoint_copy_bytes_accelerator"] for stage in report["stages"]))
        self.assertEqual(report["checkpoint_disk_bytes_per_copy"], 2 * report["coordinator_checkpoint_gather_lower_bound_bytes"])

    def test_large_meta_configuration_never_materializes_tens_of_billions_of_weights(self):
        # This is an explicit large shape test, not a published Qwen checkpoint.
        config = tiny_config(vocab_size=248320, hidden_size=2048, num_hidden_layers=40,
                             num_experts=256, num_experts_per_tok=8, moe_intermediate_size=512,
                             shared_expert_intermediate_size=512)
        report = measure_model_capacity(config, num_stages=4, mode="full", sequence_length=8)
        self.assertGreater(report["base_parameters"], 30_000_000_000)
        self.assertEqual(report["materialized_model_weight_bytes"], 0)
        # FP32 weights + gradients + Adam m/v already exceed half a terabyte.
        self.assertGreater(report["cpu_training_lower_bound_bytes"], 500_000_000_000)
        self.assertGreater(report["cpu_model_bytes_fp32"], report["structural_active_parameters"] * 4)

    def test_invalid_stage_ownership_and_sequence_fail_before_counting(self):
        with self.assertRaisesRegex(ValueError, "untied"):
            measure_model_capacity(fixture_config(tie_word_embeddings=True), 2)
        with self.assertRaises(ValueError):
            measure_model_capacity(fixture_config(), 5)
        with self.assertRaises(ValueError):
            measure_model_capacity(fixture_config(), 2, sequence_length=1)


class AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.report = measure_model_capacity(fixture_config(), 2, mode="full", sequence_length=8)

    def test_coordinator_measurement_is_mandatory(self):
        result = admit_run(self.report, [doctor("a"), doctor("b")])
        self.assertFalse(result["feasible"])
        self.assertEqual(result["status"], "requires_coordinator_measurement")
        self.assertFalse(result["training_fit_certified"])

    def test_ample_resources_only_pass_lowerbound_with_independent_copies(self):
        result = admit_run(self.report, [doctor("a"), doctor("b")], coordinator_report=doctor("c"))
        self.assertTrue(result["feasible"])
        self.assertEqual(result["status"], "lower_bound_checks_passed")
        self.assertFalse(result["training_fit_certified"])
        self.assertEqual(len(result["placements"]), 2)
        self.assertEqual(len(set(result["checkpoint_copy_hosts"])), 2)
        self.assertIn("c", result["checkpoint_copy_hosts"])
        self.assertEqual(result["checkpoint_disk_required_bytes_aggregate"], 4 * self.report["coordinator_checkpoint_gather_lower_bound_bytes"])

    def test_shared_coordinator_ram_not_counted_twice_and_30percent_reserved(self):
        stage_total = sum(s["minimum_host_bytes"] for s in self.report["stages"])
        gather = self.report["coordinator_checkpoint_gather_lower_bound_bytes"]
        # Fits workers alone but not workers + coordinator once the margin is applied.
        available = int((stage_total + gather // 2) / 0.7)
        shared = doctor("shared", ram=available)
        result = admit_run(self.report, [shared], coordinator_report=shared)
        self.assertFalse(result["feasible"])
        self.assertLess(len(result["placements"]), 2)
        self.assertEqual(len(result["host_budgets"]), 1)
        self.assertEqual(result["host_budgets"][0]["host_budget_bytes"], int(available * 0.7))

    def test_accelerator_requires_free_measurement_and_checks_vram(self):
        for available in (None, 1):
            gpu = doctor("gpu", backend="cuda", vram=available)
            result = admit_run(self.report, [gpu], coordinator_report=doctor("coordinator"))
            self.assertFalse(result["feasible"])
            self.assertEqual(len(result["placements"]), 0)
        result = admit_run(self.report, [doctor("gpu", backend="cuda", vram=10**12)], coordinator_report=doctor("coordinator"))
        self.assertTrue(result["feasible"])
        self.assertTrue(all(p["accelerator_required_bytes"] > 0 for p in result["placements"]))

    def test_distinct_directories_and_anonymous_reports_do_not_prove_replication(self):
        anonymous = doctor(None)
        result = admit_run(self.report, [anonymous, anonymous], coordinator_report=anonymous)
        self.assertFalse(result["feasible"])
        self.assertTrue(any("physical" in text for text in result["unknowns"]))
        same = doctor("same-physical-host")
        result = admit_run(self.report, [same, same], coordinator_report=same)
        self.assertFalse(result["feasible"])
        self.assertEqual(len(result["checkpoint_copy_hosts"]), 1)

    def test_repeated_measurements_do_not_double_count_same_gpu(self):
        required = max(stage["minimum_accelerator_bytes"] for stage in self.report["stages"])
        gpu = doctor("same-gpu-host", backend="cuda", vram=int(required / 0.7) + 1)
        same_gpu = {**gpu, "device": "cuda:0"}
        result = admit_run(self.report, [gpu, same_gpu], coordinator_report=doctor("coordinator"))
        self.assertFalse(result["feasible"])
        self.assertEqual(len(result["placements"]), 1)

    def test_disk_checks_each_independent_copy_and_atomic_generations(self):
        required = self.report["checkpoint_disk_bytes_per_copy"]
        result = admit_run(self.report, [doctor("worker", disk=required)], coordinator_report=doctor("coordinator"))
        # Nominal free bytes equal required still fails after 30% reserve.
        self.assertFalse(result["feasible"])
        self.assertEqual(result["checkpoint_generations_per_copy"], 2)
        with self.assertRaises(ValueError):
            admit_run(self.report, [doctor("worker")], checkpoint_copies=1)


if __name__ == "__main__":
    unittest.main()
