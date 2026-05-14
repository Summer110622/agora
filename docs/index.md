---
icon: material/home
---

# Pluralis-8B Collective Run

A decentralized pipeline-parallel pre-training run. 8B-parameter transformer, served by volunteer GPUs over the public internet.

[Get started: Agora client](quick-start/index.md){ .md-button .md-button--primary }
&nbsp;
[Agora system overview](agora-system/overview.md){ .md-button }

---

## What is Pluralis-8B?

Pluralis-8B is a collective pre-training pilot run of an 8B-parameter model using **Agora** — the system that connects a consumer GPU to a collaborative training run. Each participant holds a small portion of the model's computation graph, and allows participants to join and leave at any time; the system scales horizontally as more nodes join.

- **Consumer-grade hardware** — 24 GB GPU (e.g. RTX 4090 or equivalent), 80 GB RAM, 80 GB disk, 200 Mbps network
- **Cross-platform** — Linux and Windows + WSL2 (CUDA)
- **One-command launch** — `python3 agora.py`
- **Multi-GPU support** — run one node per GPU on the same machine
- **Live swarm participation** — join an ongoing run, warm up, then contribute gradients

<figure class="agora-figure" markdown="0">
<div class="agora-hero">
  <div class="hero-zone">
    <div class="hero-label">Discovery</div>
    <div class="hero-title">Seeds</div>
    <div class="hero-desc">Stateless DHT bootstrap. Routing only — no model data.</div>
  </div>
  <div class="hero-arrow">→</div>
  <div class="hero-zone compute">
    <div class="hero-label">Compute</div>
    <div class="hero-title">Workers</div>
    <div class="hero-desc">Pipeline-parallel stages. Forward / backward / AllReduce. Heterogeneous GPUs.</div>
  </div>
  <div class="hero-arrow">↔</div>
  <div class="hero-zone">
    <div class="hero-label">Coordination</div>
    <div class="hero-title">Trainers</div>
    <div class="hero-desc">CPU only. Microbatch routing, load balancing, data sharding.</div>
  </div>
</div>
<figcaption>Three roles, three layers. 
<a href="agora-system/training-architecture/">See the full Training Architecture for the per-stage view.</a></figcaption>
</figure>

---

## Earning points

Every node you run accrues a **score** based on the FLOPs it contributes — summed across all your peers and shown live on the [public leaderboard](https://dashboard.pluralis.ai/) as a normalized **0–100** value.

!!! info "Higher = more FLOPs"
    More uptime, better GPU, more peers running under the same account = higher rank.

[Read the full Points & Leaderboard guide](quick-start/points.md){ .md-button .md-button--primary }

---

## Research

Agora protocol learning  system design rests on four published works:

1. **Protocol Models — Scaling Decentralized Training with Communication-Efficient Model Parallelism**.&nbsp;[arXiv:2506.01260](https://arxiv.org/pdf/2506.01260). The Subspace Networks architecture and the activation-compression argument behind low-bandwidth pipeline parallelism.
2. **AsyncMesh — Fully Asynchronous Optimization for Data and Pipeline Parallelism**.&nbsp;[arXiv:2601.22442](https://arxiv.org/pdf/2601.22442). The SPARTA optimizer that lets workers in the same stage average parameters sparsely without blocking on AllReduce.
3. **Pluralis' Multi-party Training Stack**.&nbsp;[pluralis.ai/blog](https://pluralis.ai/blog/pluralis-multi-party-training-stack/). Engineering write-up tying the pieces together.
4. **SWARM Parallelism — Training Large Models Can Be Surprisingly Communication-Efficient**.&nbsp;[arXiv:2301.11913](https://arxiv.org/pdf/2301.11913). The original distributed-pipeline paper Agora builds on.