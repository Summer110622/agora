---
icon: material/flask
---

# Research Foundations { .numbered }

Two architectural innovations let Agora train large models over WAN-grade links — **Subspace Networks** compress activations in the pipeline-parallel axis; **SPARTA** compresses parameter sync in the data-parallel axis.

---

## Subspace Networks

!!! info "Paper"
    [Protocol Models: Scaling Decentralized Training with Communication-Efficient Model Parallelism](https://arxiv.org/pdf/2506.01260)

Pipeline parallelism requires high inter-layer communication. In low-bandwidth settings, this renders training infeasible without activation and activation-gradient compression.

To address this constraint, we use a class of model architecture known as Subspace Networks (SSN), designed specifically for communication-efficient distributed training. SSNs compress activations and activation gradients between otherwise unaltered transformer blocks in model-parallel stages — losslessly with respect to the model's representational capacity (the compression is an architectural property, not the wire-level state-averaging compression described in [Memory & Communication](memory-communication.md)) — achieving up to a 100x increase in communication efficiency.

SSN networks leverage the rank-collapse observed in the projection matrices of large pretrained models and constrain them to a shared and learned low-rank subspace. This reparameterization of the matrices makes the compression lossless. The architecture also handles the high-rank structure from token and positional embeddings, and exploits the recursive structure of transformers.

<figure class="agora-figure" markdown="0">
<div class="ssn-fig">
<svg viewBox="0 0 760 240" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Subspace Networks compression at a pipeline-stage boundary"><defs><marker id="ssn-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#0969da"/></marker><marker id="ssn-arrow-gray" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#999"/></marker></defs><rect x="20" y="50" width="170" height="140" rx="6" fill="#ffffff" stroke="#0969da" stroke-width="1.5"/><text x="105" y="40" text-anchor="middle" font-family="Inter,sans-serif" font-size="13" font-weight="600" letter-spacing="1" fill="#999">STAGE N</text><text x="105" y="118" text-anchor="middle" font-family="Inter,sans-serif" font-size="17" font-weight="700" fill="#111">Transformer block</text><text x="105" y="140" text-anchor="middle" font-family="Inter,sans-serif" font-size="14" fill="#777">layers L₀ … Lₖ</text><rect x="190" y="100" width="36" height="40" rx="4" fill="#f0f7ff" stroke="#0969da"/><text x="208" y="122" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" font-weight="700" fill="#0969da">SSN</text><text x="208" y="135" text-anchor="middle" font-family="Inter,sans-serif" font-size="10" fill="#0969da">↓</text><line x1="226" y1="120" x2="552" y2="120" stroke="#0969da" stroke-width="2" marker-end="url(#ssn-arrow)"/><text x="380" y="106" text-anchor="middle" font-family="Inter,sans-serif" font-size="13" font-weight="600" fill="#0969da">compressed activation</text><text x="380" y="143" text-anchor="middle" font-family="Inter,sans-serif" font-size="13" fill="#666">~ 5 Mbits / token batch</text><line x1="208" y1="155" x2="552" y2="155" stroke="#bf8700" stroke-width="2" marker-start="url(#ssn-arrow-gray)" stroke-dasharray="0"/><text x="380" y="178" text-anchor="middle" font-family="Inter,sans-serif" font-size="13" font-weight="600" fill="#bf8700">compressed activation gradient (backward)</text><rect x="552" y="100" width="36" height="40" rx="4" fill="#f0f7ff" stroke="#0969da"/><text x="570" y="118" text-anchor="middle" font-family="Inter,sans-serif" font-size="11" font-weight="700" fill="#0969da">SSN</text><text x="570" y="132" text-anchor="middle" font-family="Inter,sans-serif" font-size="10" fill="#0969da">↑</text><rect x="588" y="50" width="170" height="140" rx="6" fill="#ffffff" stroke="#0969da" stroke-width="1.5"/><text x="673" y="40" text-anchor="middle" font-family="Inter,sans-serif" font-size="13" font-weight="600" letter-spacing="1" fill="#999">STAGE N+1</text><text x="673" y="118" text-anchor="middle" font-family="Inter,sans-serif" font-size="17" font-weight="700" fill="#111">Transformer block</text><text x="673" y="140" text-anchor="middle" font-family="Inter,sans-serif" font-size="14" fill="#777">layers Lₖ₊₁ … L₂ₖ</text><text x="380" y="222" text-anchor="middle" font-family="Inter,sans-serif" font-size="16" font-weight="600" fill="#333">≈ 100× less bandwidth than the uncompressed activation (~566 Mbits)</text></svg>
</div>
<figcaption>SSN downprojects activations at each stage boundary, ships the low-rank channel across the network, and upprojects on the receiving side. The same path runs in reverse for activation gradients during the backward pass.</figcaption>
</figure>

At stage boundaries, activations are reconstructed and compressed for the forward pass, and activation gradients are reconstructed and compressed for the backward pass. As an example, a 7.5B model with a 4K embedding dimension, 32 layers, FP32 activations, and a 4K sequence length would send a 566Mbit activation for a batch size of 1. With Subspace Networks, this can be compressed to roughly 5Mbits, reducing communication time by about 100x on internet-grade links.

---

## Asynchronous Sparse Parameter Averaging

!!! info "Paper"
    [AsyncMesh: Fully Asynchronous Optimization for Data and Pipeline Parallelism](https://arxiv.org/pdf/2601.22442)

Data parallelism requires cross-worker communication in the DP axis to keep stage replicas in agreement. In a low-bandwidth setting, synchronizing the full parameter state at every step would make workers wait on the network and reintroduce communication as the bottleneck.

To address this constraint, we use asynchronous sparse parameter averaging. Workers continue processing batches and taking local optimizer steps and once every N defined local steps a small subset of parameters is averaged between peers in the same stage. This removes the synchronization barrier from the training loop, while still allowing replicas to gradually converge to a shared state.

In each averaging round, which is done every N steps (20 in the training run), only a sparse fraction of parameters is communicated (this was set to 5% for the training run). The selected subset rotates over time, so all parameters eventually get synchronized without requiring a full-model transfer in any single round.

!!! info "Production configuration"
    The values `sparse_avg=0.05` and `average_state_every=20` cited above are the settings used in the live Pluralis-8B run, configured via `worker_configs/reparam_llama/default.yaml`. Both are tunable per deployment.

This makes data-parallel communication proportional to the sparse subset size, and allows the cost of averaging to be hidden behind ongoing local computation. In practice, this is what lets workers in the same stage keep training over internet-grade links without blocking on full synchronous all-reduce rounds.
