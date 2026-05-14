---
icon: material/database
---

# Memory & Communication { .numbered }

Where parameters, gradients, and optimizer states live during training — and what crosses the network.

---

## What Lives Where

With default settings (`offload_optimizer: true`, `use_mixed_precision: true`), the optimizer states live on the CPU and the forward pass runs in BF16:

| Component | Dtype | Location (default) | Location (offload=false) |
| --- | --- | --- | --- |
| Model parameters | FP32 | GPU | GPU |
| Gradients | FP32 | GPU | GPU |
| Adam exp\_avg | FP32 | CPU | GPU |
| Adam exp\_avg\_sq | FP32 | CPU | GPU |
| Activations | FP32 | GPU | GPU |
| AllReduce buffers | FP32 | CPU | CPU |

---

## Mixed Precision

When `use_mixed_precision: true` is set, the forward pass runs under `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`. Backward ops automatically run in the same dtype as their corresponding forward ops. However, `param.grad` is always FP32 — PyTorch ensures gradient dtype matches parameter dtype, and parameters remain in FP32.

No `GradScaler` is required because BF16 has the same dynamic range as FP32 (8-bit exponent), eliminating the risk of gradient underflow.

| Stage | Data Precision | Computation Precision |
| --- | --- | --- |
| Input activations | FP32 → BF16 (autocast) | — |
| Forward matmuls | BF16 | BF16 (tensor cores) |
| Saved activations | BF16 | — |
| Backward matmuls | BF16 | BF16 (tensor cores) |
| param.grad | **FP32** | FP32 accumulation |
| Optimizer step | FP32 | FP32 |

BF16 mixed precision requires native BF16 tensor cores (compute capability ≥ 8.0): A100, H100, L40S, RTX 3090/4090, and similar. Older GPUs like T4 and V100 lack hardware BF16 support and would run slower under emulation.

---

## Activation Recomputation

<!-- TODO(eng-input): Gil flagged this section needs new prose explaining
     why we recompute activations during the backward pass. -->

Agora re-runs the forward pass during the backward pass in `ModuleCollab`. This is **not** gradient checkpointing (`torch.utils.checkpoint` is not used) — the re-forward is necessary because pipeline-stage inputs are detached between workers.

---

## Communication Patterns

Only **state averaging** (model params and optionally optimizer states) goes over the network. Gradient averaging is **local only** — the `GradientAverager` in `OptimizerAsyncSparta` never performs network communication (`allow_state_sharing` is always `False` for the gradient averager).

### What SPARTA sends during state AllReduce

| Data | Included? | Sparse? |
| --- | --- | --- |
| Model parameters (main\_parameters) | Yes | Yes (5% per round with sparse\_avg=0.05) |

Optimizer states (Adam exp\_avg, exp\_avg\_sq) are **not** averaged across peers — they stay local to each worker.

Default compression is `NoCompression` — FP32 on the wire. BF16 communication compression is available and reduces bandwidth by approximately 50%, but is not enabled by default.

### Bandwidth per round

With `sparse_avg=0.05`, each round transmits 5% of the parameter set:

* 400M params/stage: 5% × 400M × 4 bytes = **~8MB per round**
* 1B params/stage: **~20MB per round**
