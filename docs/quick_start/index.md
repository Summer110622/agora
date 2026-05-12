---
icon: rocket
order: 100
---

# Quick Start

Work through these in order:

1. **[Requirements](requirements.md)** — hardware, OS, network, and the HuggingFace token you'll need
2. **[Running Node0](run.md)** — the full run flow, what logs to expect, how to stop and restart
3. **[Advanced](advanced.md)** — multi-GPU, CLI flags, manual installation, caveats

See [Cloud Options](setup_guides/cloud.md) for running Node0 on AWS, GCP, RunPod, Tensordock, or Lambda Labs.

---

## The short version

If you [meet the requirements](requirements.md) with port **49200** open, you can start the node with:

```bash
git clone https://github.com/PluralisResearch/node0
cd node0
python3 node0_cli.py
```
The CLI will guide you through the setup process.

!!!info Check run status before joining
Live wait times are on the [Dashboard](https://dashboard.pluralis.ai/) under the **Join** button.
!!!

Progress is logged live on the [Dashboard](https://dashboard.pluralis.ai/).
