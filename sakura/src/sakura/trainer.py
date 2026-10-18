"""CPU coordinator for synchronous remote stages and consistent recovery."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from .manifest import run_hash
from .runtime import capture_rng, restore_rng


@dataclass
class StageEndpoint:
    stage_index: int
    call: Callable[[dict], dict]


class PipelineTrainer:
    def __init__(self, manifest: dict, endpoints: list[StageEndpoint], checkpoints, *, metrics_path=None,
                 restore_existing: bool = False):
        self.manifest, self.run_hash = manifest, run_hash(manifest)
        self.endpoints = sorted(endpoints, key=lambda e: e.stage_index)
        if [e.stage_index for e in self.endpoints] != list(range(manifest["num_stages"])):
            raise ValueError("Exactly one endpoint per stage is required for the synchronous baseline")
        self.checkpoints, self.step_number, self.cursor = checkpoints, 0, 0
        self.metrics_path = Path(metrics_path) if metrics_path else None
        self._checkpoint_step = None
        schemas = []
        for endpoint in self.endpoints:
            info = endpoint.call({"op": "info"})
            if info["run_hash"] != self.run_hash or info["stage_index"] != endpoint.stage_index:
                raise ValueError("Worker run/stage identity mismatch")
            if not restore_existing and (info["step"] != 0 or info["pending"]):
                raise ValueError("Worker must start clean; resume explicitly restores a committed generation")
            schemas.append(info)
        self.worker_info = schemas

    def checkpoint(self):
        states = {str(e.stage_index): e.call({"op": "snapshot"}) for e in self.endpoints}
        path = self.checkpoints.commit(states, {"step": self.step_number, "cursor": self.cursor,
                                              "rng": capture_rng()}, run_hash=self.run_hash,
                                       step=self.step_number)
        self._checkpoint_step = self.step_number
        return path

    def resume(self):
        saved = self.checkpoints.load_current(expected_run_hash=self.run_hash)
        expected = {str(i) for i in range(self.manifest["num_stages"])}
        if set(saved["stages"]) != expected:
            raise ValueError("Checkpoint does not cover the complete pipeline")
        for endpoint in self.endpoints:
            endpoint.call({"op": "restore", "state": saved["stages"][str(endpoint.stage_index)]})
        training = saved["training_state"]
        self.step_number, self.cursor = training["step"], training["cursor"]
        restore_rng(training["rng"])
        self._checkpoint_step = self.step_number
        return saved

    def train_batch(self, batch: dict) -> dict:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import load_balancing_loss_func
        if self._checkpoint_step is None:
            self.checkpoint()  # guarantees a recovery point before any mutation
        batch = {key: torch.as_tensor(value, dtype=torch.long, device="cpu") for key, value in batch.items()}
        if batch["input_ids"].ndim == 1:
            batch = {key: value.unsqueeze(0) for key, value in batch.items()}
        mask, labels = batch["attention_mask"], batch["labels"]
        if batch["input_ids"].shape != mask.shape or labels.shape != mask.shape:
            raise ValueError("Input/mask/labels shapes differ")
        target_step, start = self.step_number + 1, time.perf_counter()
        outputs, routers_per_stage, all_routers = [], [], []
        try:
            inputs = batch["input_ids"]
            for endpoint in self.endpoints:
                output = endpoint.call({"op": "forward", "step": target_step,
                                        "inputs": inputs, "attention_mask": mask,
                                        "labels": labels if endpoint.stage_index == len(self.endpoints) - 1 else None})
                routers = tuple(value.detach().requires_grad_(True) for value in output["router_logits"])
                routers_per_stage.append(routers)
                all_routers.extend(routers)
                outputs.append(output)
                inputs = output["hidden_states"]
            if outputs[-1]["loss"] is None:
                raise RuntimeError("Tail did not return a differentiable local language-model loss")
            ce = outputs[-1]["loss"].detach().requires_grad_(True)
            config = self.manifest["model"]["config"]
            config = config.get("text_config", config)
            aux = load_balancing_loss_func(tuple(all_routers), config["num_experts"],
                                           config["num_experts_per_tok"], mask)
            if not isinstance(aux, torch.Tensor):
                aux = torch.tensor(float(aux))
            loss = ce + config.get("router_aux_loss_coef", 0.001) * aux
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite global objective")
            loss.backward()
            downstream = None
            squared_norm = 0.0
            for endpoint, routers in reversed(list(zip(self.endpoints, routers_per_stage))):
                reply = endpoint.call({"op": "backward", "step": target_step,
                                       "grad_hidden": downstream,
                                       "grad_loss": ce.grad if endpoint.stage_index == len(self.endpoints) - 1 else None,
                                       "grad_routers": tuple(r.grad if r.grad is not None else torch.zeros_like(r)
                                                             for r in routers)})
                downstream = reply["grad_inputs"]
                squared_norm += reply["squared_grad_norm"]
            # No worker changes weights until the entire backward has succeeded.
            norm = math.sqrt(squared_norm)
            coefficient = min(1.0, self.manifest["training"]["max_grad_norm"] / (norm + 1e-6))
            step_results = [e.call({"op": "step", "step": target_step, "clip_coefficient": coefficient})
                            for e in self.endpoints]
            if any(result["step"] != target_step for result in step_results):
                raise RuntimeError("Worker commit version mismatch")
            self.step_number, self.cursor = target_step, self.cursor + 1
            seconds = time.perf_counter() - start
            supervised = int((labels[:, 1:] != -100).sum())
            counts = [torch.bincount(torch.topk(r.detach()[mask.reshape(-1).bool()], config["num_experts_per_tok"], dim=-1).indices.flatten(),
                                     minlength=config["num_experts"]).tolist() for r in all_routers]
            metrics = {"step": self.step_number, "cursor": self.cursor, "loss": float(loss.detach()),
                       "language_model_loss": float(ce.detach()), "router_aux_loss": float(aux.detach()),
                       "seconds": seconds, "supervised_tokens": supervised,
                       "tokens_per_second": int(mask.sum()) / max(seconds, 1e-9), "expert_counts": counts}
            if self.step_number % self.manifest["training"]["checkpoint_every"] == 0:
                self.checkpoint()
            self._log(metrics)
            return metrics
        except Exception as original:
            # A partial optimizer commit is not retried. Restore every stage.
            try:
                self.resume()
            except Exception as recovery:
                raise RuntimeError(f"Training failed ({original}); recovery also failed ({recovery}). "
                                   "Stop and resume from the last committed checkpoint after workers return.") from original
            raise

    def _log(self, metrics: dict):
        if self.metrics_path:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with self.metrics_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps({"timestamp": time.time(), "run_hash": self.run_hash, **metrics},
                                        allow_nan=False) + "\n")

    def evaluate_batch(self, batch: dict) -> dict:
        batch = {key: torch.as_tensor(value, dtype=torch.long) for key, value in batch.items()}
        if batch["input_ids"].ndim == 1:
            batch = {key: value.unsqueeze(0) for key, value in batch.items()}
        hidden = batch["input_ids"]
        for endpoint in self.endpoints:
            reply = endpoint.call({"op": "evaluate", "inputs": hidden, "attention_mask": batch["attention_mask"],
                                   "labels": batch.get("labels")})
            if reply["step"] != self.step_number:
                raise ValueError("Evaluation encountered a worker at a different checkpoint step")
            hidden = reply["hidden_states"]
        return {key: reply[key] for key in ("next_token", "nll_nats", "supervised_tokens") if key in reply}
