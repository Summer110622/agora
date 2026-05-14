---
icon: material/cloud
---

# Cloud Options

Pick a provider, deploy a GPU instance, open port `49200` for inbound TCP, and SSH in. Every section below follows the same five steps; the differences are in pricing, port-mapping behaviour, and a couple of provider-specific gotchas.

## At a glance

| Provider | Cheapest 24 GB GPU | External port | Caveats |
|---|---|---|---|
| **AWS** | NVIDIA L4 — 24 GB (`g6.8xlarge`) | Standard firewall | — |
| **GCP** | NVIDIA L4 — 24 GB (`n4-highmem-16`) | Standard firewall | — |
| **RunPod** | RTX 4090 — 24 GB | Random → set `--announce_port` | Workspace runs in Docker → use conda; `apt update` first |
| **Vast.ai** | RTX 4090 — 24 GB | Random → set `--announce_port` | Community marketplace — verify host reliability before long runs |
| **Tensordock** *(Distributed)* | RTX 4090 — 24 GB | Random → set `--announce_port` | NVIDIA toolkit + Docker need installing |
| **Lambda Labs** | NVIDIA A10 — 24 GB | Standard firewall | — |

!!! info "Network setup is shared"
    Every provider needs port `49200` reachable for inbound TCP. The provider-specific firewall / port-mapping steps live in the [Network Configuration Guide](network.md). The walkthroughs below link to the right section as you go.

---

## AWS (Amazon Web Services)

GPU options on AWS are wide; the cheapest 24 GB pick that also meets the 80 GB RAM requirement is `g6.8xlarge` (NVIDIA L4, 24 GB VRAM, 32 vCPU, 128 GB RAM).

1. **Log in** to the [AWS Management Console](https://aws.amazon.com/console/).
2. **Launch an EC2 instance**:
    - Pick an AMI with NVIDIA drivers + CUDA + PyTorch — e.g. `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04)`.
    - Pick an instance type with at least 24 GB VRAM and 80 GB RAM — e.g. `g6.8xlarge` (L4) or `g5.8xlarge` (A10G).
    - Create or select a key pair (download the `.pem` file if new).
    - Set storage to **at least 80 GB**.
3. **Open port 49200** — follow the [AWS section of the network guide](network.md#aws) to add an inbound TCP rule on the security group.
4. **SSH in**:

    ```bash
    ssh -i your-key.pem ubuntu@your-ec2-public-ip
    ```

---

## GCP (Google Cloud Platform)

Cheapest 24 GB config: an `NVIDIA L4` (24 GB VRAM) on `n4-highmem-16` (16 vCPU, 128 GB RAM).

1. **Log in** to the [Google Cloud Console](https://console.cloud.google.com/).
2. **Create a VM instance**:
    - Compute Engine → VM instances → **Create Instance**.
    - Switch from General Purpose to **GPUs**, pick `NVIDIA L4` + `n4-highmem-16`.
    - **OS and storage**: pick a CUDA + PyTorch image (e.g. `Deep Learning VM for PyTorch 2.4 with CUDA 12.4 M129`).
    - **Security** → Manage Access → add your SSH public key.
    - Click **Create**.
3. **Open port 49200** — follow the [GCP section of the network guide](network.md#gcp).
4. **SSH in** — find the external IP under instance details:

    ```bash
    ssh ubuntu@your-external-ip
    ```

---

## RunPod

Cheapest 24 GB pick: **RTX 4090** — 24 GB VRAM, ≥80 GB RAM, 8 vCPUs. RTX A5000 is a valid cheaper 24 GB alternative if 4090 availability is tight.

!!! warning "Workspace runs in Docker"
    RunPod's workspace itself is a container, which makes Docker-in-Docker awkward. **Use conda instead** — see [Useful Links → Environments](useful-links.md#environments-docker-conda).

    Most standard packages aren't preinstalled; run `apt update` before installing anything.

1. **Log in** to the [RunPod Console](https://www.runpod.io/console/home).
2. **Add an SSH key** — Settings → SSH Public Keys. If you don't have one, follow [RunPod's SSH guide](https://docs.runpod.io/pods/configuration/use-ssh).
3. **Deploy a Pod**:
    - Pick a GPU (e.g. RTX 4090).
    - Choose **Pod Template** `RunPod Pytorch 2.1` (CUDA + PyTorch preinstalled).
    - Ensure **SSH Terminal Access** is enabled.
    - Click **Deploy On-Demand**.
4. **Expose port 49200** — follow the [RunPod section of the network guide](network.md#runpod). RunPod assigns a random external port; you'll need both `--host_port 49200` and `--announce_port <external>`.
5. **SSH in** — Pod → **Connect** → **SSH** tab → use the "SSH over exposed TCP" command.

---

## Vast.ai

Cheapest pick: **RTX 4090** — 24 GB VRAM, community marketplace. RTX A5000 listings are typically cheaper but come from less reliable hosts.

!!! warning "Community hosts — verify reliability"
    Vast.ai is a marketplace of independent hosts. Prices are dramatically lower than hyperscalers but uptime and network quality vary. Check the host's reliability score (≥99%) and DLPerf number before committing to a long run. Random external port is assigned — pass both `--host_port 49200` and `--announce_port <external>`.

1. **Log in** to the [Vast.ai console](https://cloud.vast.ai/).
2. **Add an SSH key** — Account → SSH Keys.
3. **Search for a machine** — filter by GPU = `RTX 4090`, sort by price. Pick a host with reliability ≥99% and ≥200 Mbps internet.
4. **Choose a template** — `PyTorch (cuDNN Runtime)` or similar CUDA + PyTorch image. Set disk to ≥80 GB.
5. **Expose port 49200** — follow the [Vast.ai section of the network guide](network.md#vastai). Vast.ai assigns a random external port; you'll need both `--host_port 49200` and `--announce_port <external>`.
6. **Connect** — instance card → **SSH** command.

---

## Tensordock

Cheapest 24 GB pick: **RTX 4090** — 24 GB VRAM, 80 GB RAM, 8 vCPUs.

!!! warning "Distributed Compute → random external port"
    Tensordock's Distributed Compute option assigns a random external port that differs from the internal one. Pass both `--host_port 49200` and `--announce_port <external>` when you launch Agora. See the [Tensordock section of the network guide](network.md#tensordock).

1. **Log in** to the [Tensordock Deploy Dashboard](https://dashboard.tensordock.com/deploy).
2. **Add an SSH key** — Secrets → **Add Secret** → type `SSH Key`, paste your public key. New to SSH keys? See guides for [Windows](https://learn.microsoft.com/en-us/viva/glint/setup/sftp-ssh-key-gen) or [Linux](https://www.digitalocean.com/community/tutorials/how-to-configure-ssh-key-based-authentication-on-a-linux-server).
3. **Deploy a GPU**:
    - **Deploy GPU** → pick a GPU (e.g. RTX 4090).
    - Configure CPU, RAM (≥80 GB), storage (≥80 GB), location.
    - Pick OS — recommended: `Ubuntu 24.04 LTS`.
    - Click **Deploy Instance**.
4. **Configure port forwarding during provisioning** — under Port Forwarding, request port `49200`. Note the random external port assigned after deploy.
5. **Connect** — **My Servers** → click the instance → **Access** section has the SSH command.

??? note "Optional: install NVIDIA toolkit + Docker"
    Tensordock instances often ship without the NVIDIA container toolkit or Docker. To install both:

    ```bash
    sudo apt update
    sudo apt install -y nvidia-container-toolkit
    sudo systemctl restart docker
    ```

    Docker itself, if missing:

    ```bash
    sudo apt install apt-transport-https ca-certificates curl software-properties-common -y
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt update
    sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
    sudo groupadd docker
    sudo usermod -aG docker $USER
    ```

---

## Lambda Labs

Lambda sits between hyperscalers and community marketplaces — managed bare-metal GPUs with simple pricing. Cheapest 24 GB pick: **A10** (24 GB VRAM, 30 vCPUs, 200 GiB RAM).

1. **Log in** to [Lambda Cloud](https://cloud.lambda.ai/instances).
2. **Add an SSH key** — SSH Keys → add your public key.
3. **Launch an instance** — Instances → **Launch an Instance** → pick the GPU (e.g. 1× A10), pick a region and FileSystem (or create one), click **Launch**.
4. **Open port 49200** — follow the [Lambda section of the network guide](network.md#lambda-labs).
5. **SSH in** — once the instance is up, copy the command from the **SSH Login** field.
