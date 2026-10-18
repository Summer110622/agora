"""Actual Qwen3.5 tests: no synthetic substitute for the transformer model."""
import copy
import importlib.util
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

from sakura.model import (
    LoRALinear, build_model, build_stage, causal_lm_loss, forward_stages,
    inject_lora, load_lora_state_dict, lora_state_dict, merge_lora,
    export_lora_checkpoint, load_exported_lora,
    partition_model, router_aux_loss, tiny_config,
)


def fixture_config():
    return tiny_config(vocab_size=67, hidden_size=16, num_hidden_layers=4,
                       num_attention_heads=2, num_key_value_heads=1,
                       linear_num_key_heads=1, linear_num_value_heads=2,
                       moe_intermediate_size=8, shared_expert_intermediate_size=8,
                       num_experts=4, lora={"rank": 4, "alpha": 8})


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(10)
        self.ids = torch.tensor([[1, 5, 7, 9, 2, 0], [1, 3, 4, 8, 6, 2]])
        self.mask = (self.ids != 0).long()
        self.labels = self.ids.masked_fill(self.mask == 0, -100)

    def test_lora_zero_init_and_actual_updates(self):
        base = build_model(fixture_config(), mode="full", seed=7)
        reference = base(self.ids, attention_mask=self.mask, use_cache=False).logits.detach()
        inject_lora(base, rank=4, alpha=8, seed=2)
        actual = base(self.ids, attention_mask=self.mask, use_cache=False).logits
        torch.testing.assert_close(reference, actual)
        trainable = {name for name, param in base.named_parameters() if param.requires_grad}
        self.assertTrue(trainable)
        self.assertTrue(all(name.endswith(("lora_A", "lora_B")) for name in trainable))
        self.assertFalse(any("mlp.experts" in name or "mlp.gate" in name for name in trainable))
        before = lora_state_dict(base)
        optimizer = torch.optim.AdamW((p for p in base.parameters() if p.requires_grad), lr=0.01)
        for _ in range(3):
            optimizer.zero_grad()
            output = base(self.ids, labels=self.labels, attention_mask=self.mask, use_cache=False)
            output.loss.backward()
            optimizer.step()
        after = lora_state_dict(base)
        self.assertTrue(any(not torch.equal(before[n], after[n]) for n in before if n.endswith("lora_A")))
        self.assertTrue(any(not torch.equal(before[n], after[n]) for n in before if n.endswith("lora_B")))

    def test_monolithic_pipeline_loss_and_lora_gradient_parity(self):
        model = build_model(fixture_config(), seed=5)
        # Nonzero B ensures gradients exercise both factors and upstream aux paths.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name.endswith("lora_B"):
                    param.normal_(std=0.01)
        pipeline_model = copy.deepcopy(model)
        reference = model(self.ids, attention_mask=self.mask, labels=self.labels,
                          use_cache=False, output_router_logits=True)
        reference.loss.backward()
        stages = partition_model(pipeline_model, 3)
        output = forward_stages(stages, self.ids, self.mask, self.labels)
        output.loss.backward()
        torch.testing.assert_close(output.logits, reference.logits, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(output.loss, reference.loss, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(output.aux_loss, reference.aux_loss, atol=1e-6, rtol=1e-5)
        grads = dict(model.named_parameters())
        for name, param in pipeline_model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, name)
                torch.testing.assert_close(param.grad, grads[name].grad, atol=2e-6, rtol=2e-4)

    def test_detached_transport_router_gradient_matches_direct_pipeline(self):
        direct = build_model(fixture_config(), seed=11)
        remote = copy.deepcopy(direct)
        with torch.no_grad():
            for model in (direct, remote):
                for name, param in model.named_parameters():
                    if name.endswith("lora_B"):
                        param.fill_(0.02)
        reference = forward_stages(partition_model(direct, 2), self.ids, self.mask, self.labels)
        reference.loss.backward()
        stages = partition_model(remote, 2)
        head = stages[0](self.ids, self.mask)
        boundary = head.hidden_states.detach().requires_grad_(True)
        tail = stages[1](boundary, self.mask)
        router_sources = head.router_logits + tail.router_logits
        router_leaves = tuple(value.detach().requires_grad_(True) for value in router_sources)
        logits_leaf = tail.logits.detach().requires_grad_(True)
        loss = causal_lm_loss(logits_leaf, self.labels)
        loss = loss + remote.config.router_aux_loss_coef * router_aux_loss(
            router_leaves, remote.config.num_experts, remote.config.num_experts_per_tok, self.mask)
        loss.backward()
        split = len(head.router_logits)
        torch.autograd.backward((tail.logits,) + tail.router_logits,
                                (logits_leaf.grad,) + tuple(t.grad for t in router_leaves[split:]))
        torch.autograd.backward((head.hidden_states,) + head.router_logits,
                                (boundary.grad,) + tuple(t.grad for t in router_leaves[:split]))
        reference_params = dict(direct.named_parameters())
        for name, param in remote.named_parameters():
            if param.requires_grad:
                torch.testing.assert_close(param.grad, reference_params[name].grad, atol=2e-6, rtol=2e-4)

    def test_meta_owned_stage_import_conditional_namespace_and_merge_export(self):
        from safetensors.torch import save_file
        base = build_model(fixture_config(), mode="full", seed=12)
        snapshot = {name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name:
                    tensor.detach().contiguous() for name, tensor in base.state_dict().items()}
        snapshot["model.visual.unused"] = torch.ones(1)
        with tempfile.TemporaryDirectory() as folder:
            save_file(snapshot, str(Path(folder) / "model.safetensors"))
            owned = build_stage(fixture_config(), 1, 2, mode="full", pretrained_path=folder)
            self.assertFalse(hasattr(owned, "embed_tokens"))
            self.assertEqual(owned.layer_range, (2, 4))
            self.assertFalse(any(p.is_meta for p in owned.parameters()))
            original = dict(base.named_parameters())
            for local_name, param in owned.named_parameters():
                torch.testing.assert_close(param, original[owned.global_name(local_name)])
            adapted = build_model(fixture_config(), pretrained_path=folder, seed=12)
            with torch.no_grad():
                for name, param in adapted.named_parameters():
                    if name.endswith("lora_B"):
                        param.normal_(std=0.01)
            adapter = lora_state_dict(adapted)
            load_lora_state_dict(adapted, adapter)
            with self.assertRaises(ValueError):
                load_lora_state_dict(adapted, {})
            before = adapted(self.ids, use_cache=False).logits.detach()
            merged = merge_lora(adapted, inplace=False)
            self.assertFalse(any(isinstance(layer, LoRALinear) for layer in merged.modules()))
            torch.testing.assert_close(before, merged(self.ids, use_cache=False).logits)
            merged.save_pretrained(Path(folder) / "export", safe_serialization=True)
            reloaded = build_model(merged.config.to_dict(), mode="full", pretrained_path=Path(folder) / "export")
            torch.testing.assert_close(before, reloaded(self.ids, use_cache=False).logits)

    def test_owned_stage_requires_only_its_own_checkpoint_tensors(self):
        from safetensors.torch import save_file
        base = build_model(fixture_config(), mode="full", seed=19)
        owned = partition_model(base, 2)[1]
        partial = {owned.global_name(name): tensor.detach().contiguous()
                   for name, tensor in owned.named_parameters()}
        with tempfile.TemporaryDirectory() as folder:
            save_file(partial, str(Path(folder) / "model.safetensors"))
            loaded = build_stage(fixture_config(), 1, 2, pretrained_path=folder, seed=19)
            self.assertEqual(loaded.layer_range, (2, 4))
            self.assertTrue(any(param.requires_grad for param in loaded.parameters()))
            self.assertTrue(all(name.endswith(("lora_A", "lora_B"))
                                for name, param in loaded.named_parameters() if param.requires_grad))
            with self.assertRaisesRegex(ValueError, "Missing text checkpoint"):
                build_model(fixture_config(), mode="full", pretrained_path=folder)

    def test_validation(self):
        model = build_model(fixture_config())
        with self.assertRaises(ValueError):
            partition_model(model, 5)
        with self.assertRaises(ValueError):
            build_model(fixture_config(), mode="invalid")
        with self.assertRaises(ValueError):
            causal_lm_loss(torch.zeros(1, 2, 3), torch.full((1, 2), -100))

    def test_committed_adapter_export_and_owned_reload(self):
        import json
        from safetensors import safe_open
        model = build_model(fixture_config(), seed=29)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("lora_B"):
                    parameter.normal_(std=0.01)
        stages = partition_model(model, 3)
        snapshots = {str(index): {"stage_index": index, "model": stage.state_dict()}
                     for index, stage in enumerate(stages)}
        with tempfile.TemporaryDirectory() as folder:
            export_lora_checkpoint(snapshots, fixture_config(), 3, folder,
                                   "Qwen/Qwen3.5-35B-A3B-Base", "a" * 40)
            metadata = json.loads((Path(folder) / "adapter_config.json").read_text())
            self.assertEqual(metadata["r"], 4)
            self.assertEqual(metadata["lora_alpha"], 8)
            with safe_open(str(Path(folder) / "adapter_model.safetensors"), framework="pt") as reader:
                self.assertTrue(all(key.startswith("base_model.model.model.layers.") for key in reader.keys()))
                self.assertFalse(any("base_layer" in key for key in reader.keys()))
            restored = build_model(fixture_config(), seed=29)
            for stage in partition_model(restored, 3):
                load_exported_lora(stage, folder)
            torch.testing.assert_close(model(self.ids, use_cache=False).logits,
                                       restored(self.ids, use_cache=False).logits)
            with self.assertRaisesRegex(ValueError, "every committed stage"):
                export_lora_checkpoint({"0": snapshots["0"]}, fixture_config(), 3, folder, "Qwen/example", None)

    @unittest.skipUnless(importlib.util.find_spec("peft") is not None, "Optional PEFT integration dependency missing")
    def test_export_loads_in_actual_external_peft_with_logit_and_loss_parity(self):
        from peft import PeftModel
        base = build_model(fixture_config(), mode="full", seed=41)
        adapted = copy.deepcopy(base)
        inject_lora(adapted, rank=4, alpha=8, seed=41)
        with torch.no_grad():
            for name, parameter in adapted.named_parameters():
                if name.endswith("lora_B"):
                    parameter.normal_(std=0.01)
        snapshots = {str(index): stage.state_dict()
                     for index, stage in enumerate(partition_model(adapted, 3))}
        with tempfile.TemporaryDirectory() as folder:
            export_lora_checkpoint(snapshots, fixture_config(), 3, folder, "unused-local-base", None)
            external = PeftModel.from_pretrained(base, folder, is_trainable=False)
            expected = adapted(self.ids, attention_mask=self.mask, labels=self.labels, use_cache=False)
            actual = external(self.ids, attention_mask=self.mask, labels=self.labels, use_cache=False)
            torch.testing.assert_close(actual.logits, expected.logits, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(actual.loss, expected.loss, atol=1e-6, rtol=1e-5)

    def test_optin_meta_scratch_initialization_is_partition_independent_and_trains(self):
        for kind in ("hybrid", "full"):
            with self.subTest(kind=kind):
                config = fixture_config()
                config["sakura_allow_large_random"] = True
                if kind == "full":
                    config["layer_types"] = ["full_attention"] * 4
                # A one-stage reference also uses the owned meta path, not the
                # legacy whole-model RNG stream used by ordinary tiny fixtures.
                whole = build_stage(config, 0, 1, mode="full", seed=43)
                split = [build_stage(config, index, 3, mode="full", seed=43) for index in range(3)]
                expected = {whole.global_name(name): parameter for name, parameter in whole.named_parameters()}
                for stage in split:
                    self.assertIn("required_host_bytes", stage.admission_budget)
                    for name, parameter in stage.named_parameters():
                        self.assertFalse(parameter.is_meta)
                        self.assertTrue(torch.isfinite(parameter).all(), name)
                        torch.testing.assert_close(parameter, expected[stage.global_name(name)], rtol=0, atol=0)
                    for module in stage.modules():
                        if hasattr(module, "A_log"):
                            torch.testing.assert_close(module.dt_bias, torch.ones_like(module.dt_bias))
                reference = forward_stages([whole], self.ids, self.mask, self.labels)
                output = forward_stages(split, self.ids, self.mask, self.labels)
                torch.testing.assert_close(output.logits, reference.logits, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(output.loss, reference.loss, atol=1e-6, rtol=1e-5)
                before = split[-1].lm_head.weight.detach().clone()
                optimizer = torch.optim.AdamW([p for stage in split for p in stage.parameters()], lr=0.001)
                output.loss.backward()
                optimizer.step()
                self.assertFalse(torch.equal(before, split[-1].lm_head.weight))

    def test_meta_scratch_memory_gate_precedes_owned_tensor_allocation(self):
        config = fixture_config()
        config["sakura_allow_large_random"] = True
        from sakura.model import QwenStage
        with patch("sakura.model._available_host_memory", return_value=1), patch.object(QwenStage, "to_empty") as allocate:
            with self.assertRaisesRegex(MemoryError, "RAM admission"):
                build_stage(config, 0, 2, mode="full", seed=47)
            allocate.assert_not_called()
        large = fixture_config()
        large["vocab_size"] = 40_000_000
        with self.assertRaisesRegex(ValueError, "explicit sakura_allow_large_random"):
            build_stage(large, 0, 2, mode="full")


if __name__ == "__main__":
    unittest.main()
