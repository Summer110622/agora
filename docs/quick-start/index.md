---
icon: material/rocket-launch
---

# Agora Client

Work through these in order:

1. **[Requirements](requirements.md)** — hardware, OS, network, and the HuggingFace token you'll need
2. **[Running Agora](run.md)** — the full run flow, what logs to expect, how to stop and restart
3. **[Advanced](advanced.md)** — multi-GPU, CLI flags, manual installation, caveats

See [Cloud Options](setup-guides/cloud.md) for running Agora on AWS, GCP, RunPod, Tensordock, or Lambda Labs.

!!! info "Package"
    **PyPI**  `pip install pluralis-agora` 
    
    **Import**  `import agora` (or `from pluralis import agora`) 
    
    **Repo**  [`pluralis/agora`](https://github.com/PluralisResearch/agora) 

---

## The short version

If you [meet the requirements](requirements.md) with port **49200** open, you can start the node with:

```bash
git clone https://github.com/PluralisResearch/agora
cd agora
python3 agora.py
```
The CLI will guide you through the setup process.

!!! info "Check run status before joining"
    Live wait times are on the [Dashboard](https://dashboard.pluralis.ai/) under the **Join** button.

Progress is logged live on the [Dashboard](https://dashboard.pluralis.ai/).
