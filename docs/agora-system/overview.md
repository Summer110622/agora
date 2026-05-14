---
icon: material/book-open-page-variant
---

# Overview

<p class="lede">Agora is a system for <strong>multi-party pipeline-parallel training</strong> — many parties training one model together, without any single party ever holding the complete weights.</p>

No other distributed training library allows this. The property is what lets many people train models while maintaining **unextractibility** — and hence creates an avenue to collective ownership. Five technical requirements fall out of that goal:

- **Highly dynamic hardware availability** — participants join and leave mid-run
- **Extreme communication efficiency** — nodes communicate over WAN, not InfiniBand
- **Highly heterogeneous hardware** — anything from idle 4090s to legacy academic compute
- **P2P networking primitives** — fault-tolerant AllReduce without a central coordinator
- **Adversarial robustness** *(roadmap — not yet implemented)* — open participation means some peers will try to disrupt training

---

## Highly dynamic hardware availability

Training runs take a long time. In a multi-party setting we would like participants to be able to join (and leave) mid-run. This goes far beyond normal fault tolerance: the total compute available may grow or shrink, and participants may join and leave far more rapidly than GPUs fail or restart in centralised training. Rather than a stable, static block of compute, training must run on a fluid, constantly changing swarm of devices.

---

## Extreme communication efficiency

In a multi-party setting, we cannot guarantee nodes will be co-located. Forget about InfiniBand (a default requirement for large-scale training that requires a static, networked block of compute, and without which even same-region distributed training is prohibitively slow) — we cannot even guarantee nodes will be in the same datacenter. Some nodes will communicate over standard WAN.

!!! info "Bandwidth gap"
    Standard training assumes ~**1 Tbps** between nodes (InfiniBand). WAN gives you ~**100 Mbps** — several orders of magnitude slower.

---

## Highly heterogeneous hardware

We would like as many people, with whatever devices they have, to be able to contribute. This goes beyond compute heterogeneity (which alone is an unsolved problem) — we also have heterogeneity in communication bandwidth and latency. Three motivations:

**Cost.** Existing idle hardware is underutilised — you cannot use a 4090 to contribute to a big model today, yet many people have idle 4090s. This unlocks a pool of compute with very low cost (no capex, no cooling, no operating overhead beyond electricity). Important initially where we are not paying contributors.

**Broad participation.** Not just individuals — many academic departments have very limited or legacy compute. We still want them to be able to contribute.

**Protocol resiliency.** We want trained models to be guaranteed public goods with guaranteed access. The protocol must be able to run (even if degraded / slower) if all datacenter access is revoked.

---

## P2P networking primitives

NCCL, MPI, etc. all assume a single party controls the compute block. In a multi-party setting this is not the case, so fundamental communication primitives (e.g. AllReduce) must be re-implemented to be multi-party and fault-tolerant.

---

## Adversarial robustness *(roadmap)*

!!! info "Future work — not yet implemented"
    In a multi-party, open protocol, we go beyond dealing with faults — nodes may join with the intention to purposefully disrupt training. This is a known design requirement and is on the protocol-resiliency roadmap, but is **not yet implemented** in Agora today.

---

<p class="lede">Combined, this requires a completely new approach to distributed training.</p>
