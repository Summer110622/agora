---
icon: material/play-circle
---

# Running Agora

## Start the node

```bash
git clone https://github.com/PluralisResearch/agora
cd agora
python3 agora.py
```

On first run, the CLI prompts you interactively for:

- Your **HuggingFace token**
- (Optional) email address
- **GPU ID** if you have multiple GPUs
- Whether to run inside **Docker** (recommended)

Your answers are saved to a config file. Next time you run `python3 agora.py` the saved values are reused — just run it and it picks up where you left off.

---

## Docker vs Native

We recommend **Docker** — it pins the exact Python version and dependencies, isolating the node from your system Python.

If Docker isn't available, native install works but requires **Python 3.11 exactly** (not 3.10, not 3.12).

!!! info "Keep the node running after you log out"
    If you're on a remote machine, run inside [`tmux`](https://github.com/tmux/tmux/wiki) or [`screen`](https://www.gnu.org/software/screen/) so the node doesn't die when your SSH session ends:

    ```bash
    tmux new -s agora
    python3 agora.py
    # Ctrl-b d to detach; `tmux attach -t agora` to reattach
    ```

---

## What you should see

The startup sequence runs through four phases. Knowing what they look like makes debugging much faster.

### 1. Network check and weight download

```
[NETWORK]  Running internet speed test...
[DOWNLOAD] Downloading model weights...
[DOWNLOAD] Model weights downloaded. Waiting for authorization...
```

If the speed test fails or the download is too slow, your node gets dropped from the queue. Check your connection meets the 200 Mbps minimum.

### 2. Authorization queue

```
[AUTH]     Authorization queue: position 2, estimated wait: 1m
[AUTH]     Access granted for your_user
```

During high demand, the queue can be long. If your node times out waiting, just restart — it'll re-enter the queue.

### 3. Warmup (if joining an active run)

```
[WARMUP] Node warmup started, catching up with other nodes
[WARMUP] Node warmup complete
```

Warmup can take a few minutes. Your node is pulling state from peers so it can catch up to the current training step before contributing.

### 4. Training

```
[SERVER]   Training started
[TRAINING] Training step 1
[PROGRESS] Processed 51 batches in the last 60s
[PROGRESS]   Forward pass: 28 batches
[PROGRESS]   Backward pass: 23 batches
```

**Healthy signal:** `[PROGRESS] Processed [N] batches` should update every 60 seconds. As long as that keeps coming, your node is actively contributing to the swarm.

Detailed logs are written to `logs/server_gpu<ID>.log`.

---

## Verifying your contribution

Once training starts, your participation shows up on the [Dashboard](https://dashboard.pluralis.ai/). Look for your node by HuggingFace username.

---

## Stopping the node

- **Native:** press `Ctrl + C` in the terminal running the CLI
- **Docker:** `docker stop <container_name> && docker rm <container_name>`

You can rejoin anytime. Your contribution history is preserved across restarts — as long as you keep the same `private.key` file (see [Advanced → private.key](advanced.md#privatekey)).

---

## Something went wrong?

- **Error messages at startup** — check [Setup Guides](setup-guides/cloud.md), which cover auth failures, port tests, CUDA issues, and more
- **Node joined but isn't processing batches** — likely a network issue; verify port 49200 is reachable from outside
- **Ask for help** — [Zulip](https://community.pluralis.ai/#narrow/channel/26-A---Contributor-Support)

---

## Advanced
- Multiple GPUs, CLI flags, and manual installation → [Advanced](advanced.md)
