# Copyright 2026 Pluralis Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import time

from contextlib import nullcontext
from typing import Any, Dict

import torch

from hivemind.moe.server import ModuleBackend
from hivemind.utils.logging import get_logger
from hivemind.utils.nested import nested_compare, nested_flatten, nested_pack


logger = get_logger(__name__)


def _get_autocast_context(use_mixed_precision: bool, device: torch.device):
    """Return autocast context manager for BF16 mixed precision, or nullcontext if disabled."""
    if use_mixed_precision and device.type == "cuda":
        # NOTE: FP16 (which requires GradScaler) are NOT safe, don't use it here
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


class ModuleCollab(ModuleBackend):
    def __init__(
        self,
        optimizer_lock: Any,
        report_interval_minutes: float = 1,
        delay_range_forward_backward: tuple[float, float] = (0.0, 0.0),
        use_mixed_precision: bool = False,
        use_torch_compile: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.optimizer_lock = optimizer_lock
        self.delay_range_forward_backward = delay_range_forward_backward
        self.use_mixed_precision = use_mixed_precision
        self.use_torch_compile = use_torch_compile

        if use_torch_compile:
            self._compiled_module = torch.compile(self.module, dynamic=True)
            logger.info("torch.compile enabled (regional: forward/backward call-sites only)")

        # Gradient reporting configuration
        self.report_interval_minutes = report_interval_minutes
        self.report_interval_seconds = report_interval_minutes * 60
        self.last_report_time = time.time()
        self.grad_norms = []  # Store accumulated gradient norms

        if self.use_mixed_precision:
            logger.info("Mixed precision (BF16) enabled for forward/backward passes")

    def backward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply backward pass to an aggregated batch of requests.

        Used by Runtime, do not call this manually.
        To submit a request for asynchronous processing, please use ``ModuleBackend.backward_pool.submit_task``.

        Subclassing:
           - This method receives a sequence of torch tensors following ``nested_flatten(self.backward_schema)``;

           - It should return gradients w.r.t. inputs that follow ``nested_flatten(self.forward_schema)``;

           - Runtime doesn't guarantee that backward will be performed in the same order and for the same data
           as forward, so we recommend stateless backward pass that re-runs expert forward pass inside backward.

           - Please make sure to call ``ModuleBackend.on_backward`` after each call to backward
        """
        (args, kwargs), grad_outputs = nested_pack(inputs, structure=self.backward_schema)

        with torch.enable_grad():
            with self.optimizer_lock:
                args = [
                    tensor.detach().requires_grad_(True) if tensor.is_floating_point() else tensor.detach()
                    for tensor in args
                ]
                kwargs = {
                    input_key: (
                        tensor.detach().requires_grad_(True) if tensor.is_floating_point() else tensor.detach()
                    )
                    for input_key, tensor in kwargs.items()
                }

                batch_size = args[0].size(0)
                device = args[0].device

                # Use BF16 autocast for forward pass only (per PyTorch AMP recommendations)
                # Backward ops automatically run in the same dtype as their corresponding forward ops
                # param.grad is always FP32 (matches param.dtype)
                with _get_autocast_context(self.use_mixed_precision, device):
                    _module = self._compiled_module if self.use_torch_compile else self.module
                    outputs = _module(*args, **kwargs)

                assert nested_compare(outputs, grad_outputs), "outputs and grad_outputs must have the same structure"

                outputs_flat = tuple(nested_flatten(outputs))

                grad_outputs_flat = tuple(
                    map(
                        lambda grad, out: grad.to(device=out.device, dtype=out.dtype, non_blocking=True),
                        nested_flatten(grad_outputs),
                        outputs_flat,
                    )
                )

                torch.autograd.backward(
                    outputs_flat, grad_tensors=grad_outputs_flat, create_graph=False, retain_graph=False
                )

            self.on_backward(batch_size)

        # Delay used to simulate latency
        if self.delay_range_forward_backward[1] > 0:
            time.sleep(random.uniform(self.delay_range_forward_backward[0], self.delay_range_forward_backward[1]))

        return tuple(
            x.grad if isinstance(x.grad, torch.Tensor) else torch.zeros_like(x) for x in nested_flatten((args, kwargs))
        )

    def on_backward(self, batch_size: int) -> None:
        """Train the expert for one step.

        This method is called by ``ModuleBackend.backward`` after computing gradients.
        """
        # Logging grad norms
        grads = [p.grad for p in self.module.parameters() if p.grad is not None]

        # Safety check: gradients must be FP32 for stable accumulation and optimizer step
        # FP16 gradients risk underflow - only BF16 autocast (same dynamic range as FP32) is safe.
        if grads:
            assert grads[0].dtype == torch.float32, f"Expected FP32 gradients, got {grads[0].dtype}"

        self.grad_norms.append(torch.nn.utils.get_total_norm(grads).item())

        if self.optimizer is not None:
            self.optimizer.step(batch_size=batch_size)
            self.optimizer.zero_grad(set_to_none=not self.use_torch_compile)

            if self.scheduler is not None:
                self.scheduler.step()

        self._maybe_report_grad_stats()

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Overrides `ModuleBackend.forward` to add a delay and optional mixed precision."""
        # Delay used to simulate latency
        if self.delay_range_forward_backward[1] > 0:
            time.sleep(random.uniform(self.delay_range_forward_backward[0], self.delay_range_forward_backward[1]))

        # Determine device from first input tensor
        device = inputs[0].device if inputs else torch.device("cpu")

        # Use BF16 autocast for forward pass if mixed precision is enabled
        with self.optimizer_lock:
            with _get_autocast_context(self.use_mixed_precision, device):
                if self.use_torch_compile:
                    args, kwargs = nested_pack(inputs, structure=self.forward_schema)
                    with torch.no_grad():
                        outputs = self._compiled_module(*args, **kwargs)
                    return tuple(nested_flatten(outputs))
                return super().forward(*inputs)

    def _maybe_report_grad_stats(self):
        """Check if enough time has passed and report gradient statistics if so."""
        current_time = time.time()

        if current_time - self.last_report_time >= self.report_interval_seconds:
            if self.grad_norms:
                max_grad = max(self.grad_norms)
                avg_grad = sum(self.grad_norms) / len(self.grad_norms)

                logger.info(f"Pre Max total_grad: {max_grad:.6f}")
                logger.info(f"Pre Avg total_grad: {avg_grad:.6f}")

                # Reset for next interval
                self.grad_norms.clear()
            else:
                logger.info(f"No gradient norms recorded in the last {self.report_interval_minutes} minutes")

            self.last_report_time = current_time

    def get_info(self) -> Dict[str, Any]:
        """Get expert parameters and stats. Used by RemoteExpert to check shapes and for DMoE orchestration."""
        info = super().get_info()
        if hasattr(self.module, "model_args"):
            info["model_args"] = dict(self.module.model_args)

        return info
