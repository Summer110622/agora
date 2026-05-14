---
icon: material/tools
---

# Setup Guides

Reference material that's linked from the main quick-start flow but doesn't belong inline. Pull from these when you hit the relevant step — they aren't meant to be read top-to-bottom.

---

## Cloud Options

If you don't have a 24 GB GPU locally, rent one. The guide picks the cheapest practical instance per provider and walks through provisioning + the provider-specific port-mapping gotchas.

| Provider | Cheapest 24 GB GPU |
|---|---|
| **AWS** | NVIDIA L4 (`g6.8xlarge`) |
| **GCP** | NVIDIA L4 (`n4-highmem-16`) |
| **RunPod** | RTX 4090 |
| **Vast.ai** | RTX 4090 |
| **Tensordock** | RTX 4090 |
| **Lambda Labs** | A10 |

→ [Read the full Cloud Options guide](cloud.md)

---

## Network Configuration

Agora needs inbound TCP on port `49200` (one port per GPU). How you open it depends on where the node lives:

- **Cloud VM** — security group / firewall rule on the provider
- **Laptop / home machine** — local firewall **and** router port forwarding
- **WSL2** — extra port-proxy step

RunPod and Tensordock remap the port externally to a random number — you'll need `--host_port 49200` and `--announce_port <external>` when launching.

→ [Read the full Network Configuration guide](network.md)

---

## Useful Links

External references for the toolchain Agora sits on top of — SSH keys, Docker, conda, NVIDIA drivers, HuggingFace tokens, Linux basics. Linked from the main guides where relevant; collected here for browsing.

→ [Read the full Useful Links page](useful-links.md)
