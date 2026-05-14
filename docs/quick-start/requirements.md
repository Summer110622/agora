---
icon: material/format-list-checks
---

# Requirements

Your machine needs to meet these baselines before joining an Agora swarm. Run through each section.

!!! info "Don't have a machine that meets the requirements?"
    You can rent a GPU instance from AWS, GCP, RunPod, Vast.ai, Tensordock, or Lambda Labs. The [Cloud Options guide](setup-guides/cloud.md) walks through picking and provisioning one for each provider, including cheapest-option recommendations.

---

## Hardware

| Resource | Minimum | Notes |
|----------|---------|-------|
| **GPU** | 24 GB VRAM | RTX 4090 or equivalent. Consumer cards work. |
| **RAM** | 80 GB per GPU | A 2-GPU machine needs 160 GB total. |
| **Disk** | 80 GB free | For model weights, checkpoints, logs. |
| **Network** | 200 Mbps stable | See shared-bandwidth note below. |

!!! warning "Cloud bandwidth is often shared"
    Some cloud providers advertise bandwidth shared across tenants. Your effective throughput may be significantly lower than the headline number. Run a speed test on the actual instance before committing to a long run.

Check your GPU is visible:

```bash
nvidia-smi
```

You should see your card and its VRAM. If this command isn't found, install [NVIDIA drivers](setup-guides/useful-links.md#gpu-nvidia-drivers) first.

---

## Operating system

Agora runs on:

- **Linux** — any modern distro with recent NVIDIA drivers
- **Windows 10/11 + WSL2** — [enable CUDA support in WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

macOS is not supported (no CUDA).

---

## Network

Other peers need to reach your node on a TCP port. The default is **49200**. You need this port **open for inbound connections**.

Where you need to configure it depends on your setup:

- **Cloud VM** — open the port in your provider's security group / firewall rule. See the [cloud provider walkthrough](setup-guides/cloud.md) for AWS, GCP, RunPod, Tensordock, and Lambda Labs.
- **Laptop / home machine** — open the port on both your local firewall **and** your router (port forwarding). See the [network configuration guide](setup-guides/network.md).
- **WSL2** — requires an extra port-proxy step. See the [WSL section of the network guide](setup-guides/network.md#wsl2).

!!! info "Multiple GPUs need multiple ports"
    Each GPU on the same machine needs its own port — 49200 for GPU 0, 49201 for GPU 1, and so on. Open each one.

!!! warning "RunPod and Tensordock remap ports externally"
    These providers assign a random external port that differs from the internal one. You'll need to pass both via `--host_port` and `--announce_port`. The [network guide](setup-guides/network.md#runpod) walks through finding your external port.

---

## Authentication — HuggingFace token

You need a free HuggingFace access token. It's used for identity only — no read/write scopes required.

1. Create an account at [huggingface.co](https://huggingface.co) if you don't have one
2. [Generate an access token](https://huggingface.co/docs/hub/en/security-tokens)
3. Keep it handy — the CLI will prompt for it on first run

---

## Pre-flight checklist

Before moving on, confirm:

- [ ] GPU has ≥ 24 GB VRAM (verified with `nvidia-smi`)
- [ ] 80 GB RAM per GPU, 80 GB free disk
- [ ] Stable 200 Mbps+ connection
- [ ] Port 49200 open for inbound TCP (per GPU)
- [ ] HuggingFace token generated and saved somewhere you can paste from

Once all of these are ticked, [continue to running Agora](run.md).
