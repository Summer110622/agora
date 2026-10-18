"""Synchronous, versioned stage workers. No optimizer update occurs in backward."""
from __future__ import annotations

import hashlib
import random
import threading

import torch

from .doctor import resolve_device
from .manifest import run_hash


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def capture_rng() -> dict:
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = [x.cpu() for x in torch.cuda.get_rng_state_all()]
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state().cpu()
    return state


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


class StageWorker:
    def __init__(self, manifest: dict, stage_index: int, device: str = "cpu", *, base_path: str | None = None):
        from .model import build_stage
        self.manifest, self.run_hash = manifest, run_hash(manifest)
        if not 0 <= stage_index < manifest["num_stages"]:
            raise ValueError("Stage index out of range")
        self.stage_index, self.device = stage_index, resolve_device(device)
        self.lock, self.step_number = threading.RLock(), 0
        source = manifest["model"]
        if source.get("pretrained_path"):
            from .artifacts import verify_snapshot
            verify_snapshot(base_path or source["pretrained_path"], source["artifacts"])
        self.stage = build_stage(source["config"], stage_index, manifest["num_stages"], mode=manifest["mode"],
                                 pretrained_path=base_path or source.get("pretrained_path"),
                                 revision=source["revision"] if source.get("pretrained_path") else None,
                                 seed=source.get("seed", 0), device=str(self.device), dtype=torch.float32)
        self.stage.train()
        self.trainable = [p for p in self.stage.parameters() if p.requires_grad]
        if not self.trainable:
            raise ValueError("Every training stage must own trainable parameters")
        self.offloaded = self.device.type != "cpu" or any(p.dtype != torch.float32 for p in self.trainable)
        self.optimizer_parameters = ([torch.nn.Parameter(p.detach().float().cpu().clone()) for p in self.trainable]
                                     if self.offloaded else self.trainable)
        self.optimizer = torch.optim.AdamW(self.optimizer_parameters, lr=manifest["training"]["learning_rate"])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)
        self.pending = None
        self._last_backward = None

    def handle(self, request: dict) -> dict:
        with self.lock:
            operation = request.get("op")
            if operation == "probe":
                payload = request.get("payload", "")
                if not isinstance(payload, str) or len(payload) > 1024 * 1024:
                    raise ValueError("Invalid probe payload")
                return {"payload": payload}
            if operation == "info":
                names = [(n, list(p.shape), str(p.dtype)) for n, p in self.stage.named_parameters() if p.requires_grad]
                return {"run_hash": self.run_hash, "stage_index": self.stage_index, "step": self.step_number,
                        "device": str(self.device), "trainable_parameters": sum(p.numel() for p in self.trainable),
                        "parameter_schema": hashlib.sha256(repr(names).encode()).hexdigest(),
                        "pending": self.pending is not None}
            if operation == "snapshot":
                return self.snapshot()
            if operation == "restore":
                self.restore(request["state"])
                return {"step": self.step_number}
            if operation == "evaluate":
                if self.pending is not None:
                    raise RuntimeError("Cannot evaluate during an in-flight training microbatch")
                self.stage.eval()
                try:
                    with torch.no_grad():
                        mask = request["attention_mask"].to(self.device)
                        output = self.stage(request["inputs"].to(self.device), attention_mask=mask)
                        reply = {"hidden_states": cpu_tree(output.hidden_states), "step": self.step_number}
                        if output.logits is not None:
                            lengths = mask.sum(-1).long() - 1
                            if (lengths < 0).any():
                                raise ValueError("Cannot generate from an empty prompt")
                            reply["next_token"] = cpu_tree(output.logits[
                                torch.arange(mask.shape[0], device=self.device), lengths].argmax(-1))
                            if request.get("labels") is not None:
                                labels = request["labels"].to(self.device)[:, 1:]
                                count = int((labels != -100).sum())
                                nll = torch.nn.functional.cross_entropy(
                                    output.logits[:, :-1].reshape(-1, output.logits.shape[-1]),
                                    labels.reshape(-1), ignore_index=-100, reduction="sum")
                                reply.update({"nll_nats": float(nll), "supervised_tokens": count})
                        return reply
                finally:
                    self.stage.train()
            expected_step = request.get("step")
            if type(expected_step) is not int:
                raise ValueError("Training operation requires integer step")
            if operation == "step" and expected_step == self.step_number:
                return {"step": self.step_number, "already_applied": True}
            if expected_step != self.step_number + 1:
                raise ValueError(f"Stale/out-of-order step: expected {self.step_number + 1}, got {expected_step}")
            if operation == "forward":
                if self.pending is not None:
                    raise RuntimeError("Previous microbatch still pending; restore a consistent checkpoint before retry")
                self.optimizer.zero_grad(set_to_none=True)
                self.stage.zero_grad(set_to_none=True)
                inputs = request["inputs"].to(self.device).detach()
                if inputs.is_floating_point():
                    inputs.requires_grad_(True)
                mask = request["attention_mask"].to(self.device)
                positions = request.get("position_ids")
                if positions is not None:
                    positions = positions.to(self.device)
                with torch.enable_grad():
                    output = self.stage(inputs, attention_mask=mask, position_ids=positions)
                    loss = None
                    if output.logits is not None and request.get("labels") is not None:
                        labels = request["labels"].to(self.device)
                        if not (labels[:, 1:] != -100).any():
                            raise ValueError("Batch has no supervised next-token targets")
                        loss = torch.nn.functional.cross_entropy(
                            output.logits[:, :-1].reshape(-1, output.logits.shape[-1]),
                            labels[:, 1:].reshape(-1), ignore_index=-100)
                self.pending = {"inputs": inputs, "output": output, "loss": loss, "backward_done": False}
                return {"hidden_states": cpu_tree(output.hidden_states),
                        "logits": cpu_tree(output.logits) if output.logits is not None and loss is None else None,
                        "loss": cpu_tree(loss) if loss is not None else None,
                        "router_logits": cpu_tree(output.router_logits)}
            if operation == "backward":
                if self.pending is None:
                    raise RuntimeError("No matching forward")
                if self.pending["backward_done"]:
                    return self._last_backward
                output = self.pending["output"]
                pairs = [(output.hidden_states, request.get("grad_hidden")),
                         (output.logits, request.get("grad_logits")),
                         (self.pending["loss"], request.get("grad_loss"))]
                router_grads = request.get("grad_routers", ())
                if len(router_grads) != len(output.router_logits):
                    raise ValueError("Router gradient schema mismatch")
                pairs += list(zip(output.router_logits, router_grads))
                outputs, gradients = [], []
                for tensor, gradient in pairs:
                    if tensor is not None and gradient is not None and tensor.requires_grad:
                        if tensor.shape != gradient.shape or not torch.isfinite(gradient).all():
                            raise ValueError("Invalid/non-finite boundary gradient")
                        outputs.append(tensor)
                        gradients.append(gradient.to(device=tensor.device, dtype=tensor.dtype))
                if not outputs:
                    raise ValueError("No differentiable output received a gradient")
                torch.autograd.backward(outputs, gradients)
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.trainable):
                    raise FloatingPointError("Non-finite parameter gradient; no optimizer step performed")
                self.pending["backward_done"] = True
                gradient = self.pending["inputs"].grad
                squared_norm = sum(float(p.grad.detach().cpu().double().square().sum())
                                   for p in self.trainable if p.grad is not None)
                self._last_backward = {"grad_inputs": cpu_tree(gradient) if gradient is not None else None,
                                       "squared_grad_norm": squared_norm}
                return self._last_backward
            if operation == "step":
                if self.pending is None or not self.pending["backward_done"]:
                    raise RuntimeError("Optimizer step requires a complete backward")
                coefficient = float(request.get("clip_coefficient", 1.0))
                if not 0 < coefficient <= 1:
                    raise ValueError("Invalid global clipping coefficient")
                for parameter in self.trainable:
                    if parameter.grad is not None:
                        parameter.grad.mul_(coefficient)
                if self.offloaded:
                    for parameter, master in zip(self.trainable, self.optimizer_parameters):
                        master.grad = parameter.grad.detach().float().cpu() if parameter.grad is not None else None
                self.optimizer.step()
                if self.offloaded:
                    with torch.no_grad():
                        for parameter, master in zip(self.trainable, self.optimizer_parameters):
                            parameter.copy_(master.to(device=parameter.device, dtype=parameter.dtype))
                self.scheduler.step()
                self.step_number = expected_step
                self.pending = self._last_backward = None
                return {"step": self.step_number}
            raise ValueError(f"Unknown worker operation: {operation}")

    def snapshot(self) -> dict:
        if self.pending is not None:
            raise RuntimeError("Cannot checkpoint an in-flight microbatch")
        adapter_only = self.manifest["mode"] == "lora" and bool(self.manifest["model"].get("pretrained_path"))
        state = self.stage.state_dict()
        if adapter_only:
            state = {name: value for name, value in state.items() if "lora_A" in name or "lora_B" in name}
        return {"step": self.step_number, "run_hash": self.run_hash, "stage_index": self.stage_index,
                "mode": self.manifest["mode"], "adapter_only": adapter_only,
                "model": cpu_tree(state), "optimizer": cpu_tree(self.optimizer.state_dict()),
                "optimizer_master": cpu_tree(self.optimizer_parameters) if self.offloaded else None,
                "scheduler": self.scheduler.state_dict(), "rng": capture_rng()}

    def restore(self, state: dict) -> None:
        if (state.get("run_hash"), state.get("stage_index"), state.get("mode")) != (
                self.run_hash, self.stage_index, self.manifest["mode"]):
            raise ValueError("Checkpoint run/stage/training-mode mismatch")
        adapter_only = self.manifest["mode"] == "lora" and bool(self.manifest["model"].get("pretrained_path"))
        if state.get("adapter_only") != adapter_only:
            raise ValueError("Checkpoint model-state kind mismatch")
        expected = self.stage.state_dict()
        if adapter_only:
            expected = {k: v for k, v in expected.items() if "lora_A" in k or "lora_B" in k}
        if set(expected) != set(state["model"]) or any(expected[k].shape != state["model"][k].shape for k in expected):
            raise ValueError("Checkpoint parameter schema mismatch")
        self.stage.load_state_dict(state["model"], strict=not adapter_only)
        if self.offloaded:
            masters = state.get("optimizer_master")
            if masters is None and all(p.dtype == torch.float32 for p in self.trainable):
                # CPU checkpoints use the FP32 model itself as the master copy.
                # A resumed CUDA/ROCm/MPS worker must not require absent duplicates.
                masters = [p.detach().float().cpu() for p in self.trainable]
            if not isinstance(masters, list) or len(masters) != len(self.optimizer_parameters):
                raise ValueError("Missing CPU optimizer master parameters")
            with torch.no_grad():
                for master, saved in zip(self.optimizer_parameters, masters):
                    if master.shape != saved.shape:
                        raise ValueError("CPU optimizer master parameter schema mismatch")
                    master.copy_(saved)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.optimizer.zero_grad(set_to_none=True)
        restore_rng(state["rng"])
        self.step_number = int(state["step"])
        self.pending = self._last_backward = None
