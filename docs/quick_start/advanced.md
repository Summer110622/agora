---
icon: gear
order: 70
---

# Advanced

---

## Multiple GPUs

Run one CLI instance per GPU:

```bash
python3 node0_cli.py --gpu_id 0
python3 node0_cli.py --gpu_id 1
```

Each GPU automatically uses its own port (49200, 49201, ...). Make sure every port is open for inbound TCP.

- **Native** — run each command in a separate terminal (or tmux window)
- **Docker** — each instance runs in its own container automatically

---

## CLI reference

The CLI prompts interactively, but every value can be passed as a flag:

| Flag | Description |
|------|-------------|
| `--gpu_id <ID>` | GPU to use (default: 0) |
| `--token <TOKEN>` | HuggingFace token |
| `--email <EMAIL>` | Email address (optional) |
| `--host_port <PORT>` | Listening port (default: `49200 + gpu_id`) |
| `--announce_port <PORT>` | External port if different from `--host_port` (e.g. RunPod, Tensordock) |
| `--use_docker` | Run inside Docker |
| `--log_file <PATH>` | Log file path (default: `logs/server_gpu<ID>.log`) |
| `--identity_path <PATH>` | Identity key path (default: `private_gpu<ID>.key`) |
| `--skip_input` | Non-interactive — all values must come from flags or saved config |
| `--reconfigure` | Re-prompt every setting from scratch |

The `--log_file` and `--identity_path` defaults are almost always what you want — only override if you have a specific reason.

---

## Manual installation

### 1. Install dependencies

**Docker:**

```bash
docker build . -t pluralis_node0 --label image_version=1
```

**Native (no Docker):**

Prerequisites: Python 3.11, `pip >= 25.3`, NVIDIA drivers with CUDA 12.8, conda.

```bash
# Create and activate a Python 3.11 environment
conda create -y -n node0 python=3.11
conda activate node0

# Upgrade pip
pip install --upgrade "pip>=25.3"

# Install PyTorch with CUDA 12.8 support
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Install the two required packages from source
pip install --build-constraint constraints.txt -e ./node0_server
pip install --build-constraint constraints.txt -e ./node0
```

### 2. Read runtime parameters

`run.json` at the repo root contains values required by `run_server.py`:

| `run.json` field | `run_server.py` argument |
|------------------|--------------------------|
| `run_config` | `--config` |
| `auth_server` | `--auth_server` |
| `prom_gateway` | `--prom_gateway` |
| `seeds` (array) | `--initial_peers` |

```bash
RUN_CONFIG=$(jq -r '.run_config' run.json)
AUTH_SERVER=$(jq -r '.auth_server' run.json)
PROM_GATEWAY=$(jq -r '.prom_gateway' run.json)
SEEDS=$(jq -r '.seeds[]' run.json)
```

### 3. Get your public IP

```bash
PUBLIC_IP=$(curl -s https://api.ipify.org)
```

### 4. Launch

Default port convention: `49200 + gpu_id`. Each GPU needs its own port.

**Native:**

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python3.11 node0/src/node0/run_server.py \
  --gpu_id <gpu_id> \
  --config "$RUN_CONFIG" \
  --token <hf_token> \
  --auth_server "$AUTH_SERVER" \
  --prom_gateway "$PROM_GATEWAY" \
  --host_maddrs /ip4/0.0.0.0/tcp/<port> \
  --announce_maddrs /ip4/$PUBLIC_IP/tcp/<port> \
  --initial_peers $SEEDS \
  --email <email> \
  --log_file logs/server_gpu<gpu_id>.log \
  --identity_path private_gpu<gpu_id>.key
```

**Docker:**

```bash
docker run -d --name node0_gpu<gpu_id> --ipc=host --network=host \
  --gpus device=<gpu_id> \
  -v $(pwd):/home -w /home \
  pluralis_node0 \
  bash -c "CUDA_VISIBLE_DEVICES=0 python3.11 node0/src/node0/run_server.py \
    --gpu_id <gpu_id> \
    --config $RUN_CONFIG \
    --token <hf_token> \
    --auth_server $AUTH_SERVER \
    --prom_gateway $PROM_GATEWAY \
    --host_maddrs /ip4/0.0.0.0/tcp/<port> \
    --announce_maddrs /ip4/$PUBLIC_IP/tcp/<port> \
    --initial_peers $SEEDS \
    --email <email> \
    --log_file logs/server_gpu<gpu_id>.log \
    --identity_path private_gpu<gpu_id>.key"
```

---

## Caveats

### private.key

On first run, Node0 generates a `private.key` file (or `private_gpu<ID>.key` for multi-GPU setups). This is your node's cryptographic identity.

- **Required** for secure communication within the swarm
- **Keep it private** — never share or commit it
- **Tied to your HuggingFace account** — if you try to rejoin with the same `private.key` but a different HF account, authorization fails with `This peer_id is already used by another user`
- **Losing it** means you'll join as a new identity — previous swarm contributions remain on the dashboard, but your node_id changes

### Docker file ownership

Files created inside a Docker container are owned by the container's user, not the host user. To modify or delete them from the host:

**Linux:**

```bash
sudo chown -R $USER <path/to/project>
```

This commonly matters for `logs/` and checkpoint files written during a Docker run.
