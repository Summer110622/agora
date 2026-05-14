---
icon: material/play-circle
---

# Startup Sequence

What happens when you launch a local Agora run.

The startup flows through a chain of shell scripts: `local_run.sh` calls `setup_containers.sh` to provision Docker infrastructure, then `launch_agora.sh` execs into the container and runs `tmux_launcher.sh` to start all Agora processes.

---

## Default Configuration

```text
CFG = llama/debug_sparta_async_C.yaml
DHT = 3          # 3 DHT instances per seed
STAGES = 2       # head + tail (no body stages)
PIPES = 2        # 2 workers per stage = 4 workers total
TRAINERS = 2     # 2 trainer processes
TARGET_BS = 64   # target batch size per epoch
GPUS = 1         # GPU count for round-robin assignment
```

---

## Infrastructure Containers

The `setup_containers.sh` script starts several Docker containers on the host network. Only one is required for training with auth disabled:

* **`agora`** — The main container where all Agora processes run. Mounts the entire repo at `/home/pluralis/dev` and has `--gpus all`.

Additional containers (only needed with `--use_auth True` or for monitoring):

* `postgres` — DB backend for authorization and health monitoring
* `redis` — Rate limiting and hardware threshold checks
* `prometheus` / `pushgateway` — Metrics scraping (ports 9090/9091)
* `adminer` — Database web UI

---

## Process Startup Order

All processes run inside a single Docker container in separate tmux panes. The minimum set for training with auth disabled is: **Seeds → wait → Workers → Trainers**.

| Order | Process | tmux Session | Required? | Waits For |
| --- | --- | --- | --- | --- |
| 1 | Prometheus config | (inline) | No (monitoring) | Nothing |
| 2 | Authorizer | `auth` | Only with use\_auth | Nothing (sleep 5 after) |
| 3 | Seed 0 | `seed` pane 0 | **Yes** | Nothing |
| 4 | Seed 1 | `seed` pane 1 | **Yes** | Nothing |
| 5 | **WAIT** | — | **Yes** | Seed multiaddr files |
| 6 | Workers (4 total) | `workers` | **Yes** | Seed addrs on disk |
| 7 | Monitors (2) | `trainers` pane 0-1 | No (monitoring) | Nothing |
| 8 | Trainers (2) | `trainers` pane 2-3 | **Yes** | Nothing |

---

## Port Allocation

Each worker receives a deterministic port and GPU assignment based on its index. For a default 4-worker setup (2 stages, 2 pipes):

| Worker | Stage | Pipe | Port | Role |
| --- | --- | --- | --- | --- |
| Worker 0 | Head | 0 | 12000 | Seed peer (init from scratch) |
| Worker 1 | Tail | 0 | 12010 | Seed peer (init from scratch) |
| Worker 2 | Head | 1 | 12020 | Downloads state from peers |
| Worker 3 | Tail | 1 | 12030 | Downloads state from peers |

*Ports are allocated by `tmux_launcher.sh`, not enforced server-side — the formula `12000 + 10 * worker_index` is a launch-script convention.*

---

## Inter-Process Communication

<figure class="agora-figure" markdown="0">
<div class="ipc-fig">
<svg viewBox="0 0 760 440" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Inter-process communication sequence diagram — Seeds, Workers, Trainers">
  <defs>
    <marker id="ipc-arr-disc" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#1a7f37"/></marker>
    <marker id="ipc-arr-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#0969da"/></marker>
    <marker id="ipc-arr-warn" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#bf8700"/></marker>
    <marker id="ipc-arr-gray" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#666"/></marker>
  </defs>

  <!-- Actor header boxes -->
  <rect x="60" y="20" width="120" height="44" rx="6" fill="#ffffff" stroke="#0969da" stroke-width="1.5"/>
  <text x="120" y="48" text-anchor="middle" font-family="Inter,sans-serif" font-size="14" font-weight="700" fill="#111">Seeds</text>

  <rect x="320" y="20" width="120" height="44" rx="6" fill="#ffffff" stroke="#0969da" stroke-width="1.5"/>
  <text x="380" y="48" text-anchor="middle" font-family="Inter,sans-serif" font-size="14" font-weight="700" fill="#111">Workers</text>

  <rect x="580" y="20" width="120" height="44" rx="6" fill="#ffffff" stroke="#0969da" stroke-width="1.5"/>
  <text x="640" y="48" text-anchor="middle" font-family="Inter,sans-serif" font-size="14" font-weight="700" fill="#111">Trainers</text>

  <!-- Lifelines (dashed vertical lines) -->
  <line x1="120" y1="64" x2="120" y2="420" stroke="#bbb" stroke-width="1" stroke-dasharray="4 4"/>
  <line x1="380" y1="64" x2="380" y2="420" stroke="#bbb" stroke-width="1" stroke-dasharray="4 4"/>
  <line x1="640" y1="64" x2="640" y2="420" stroke="#bbb" stroke-width="1" stroke-dasharray="4 4"/>

  <!-- "time ↓" gutter on left margin -->
  <text x="22" y="90" font-family="Inter,sans-serif" font-size="10" font-weight="600" letter-spacing="1" fill="#999">TIME</text>
  <line x1="30" y1="98" x2="30" y2="405" stroke="#ccc" stroke-width="1"/>
  <path d="M26 400 L30 410 L34 400 z" fill="#ccc"/>

  <!-- 1. Seeds → Workers: read multiaddr file (green arrow above the write self-loop) -->
  <line x1="120" y1="95" x2="378" y2="95" stroke="#1a7f37" stroke-width="1.5" stroke-dasharray="6 4" marker-end="url(#ipc-arr-disc)"/>
  <text x="249" y="87" text-anchor="middle" font-family="Inter,sans-serif" font-size="12" font-weight="600" fill="#1a7f37">read multiaddr file (via disk)</text>

  <!-- 2. Seeds self-action: write multiaddress files -->
  <path d="M120 135 L 170 135 L 170 155 L 126 155" fill="none" stroke="#666" stroke-width="1.5" marker-end="url(#ipc-arr-gray)"/>
  <text x="180" y="148" font-family="Inter,sans-serif" font-size="12" fill="#444">write <tspan font-family="ui-monospace,Menlo,monospace" font-size="11">multiaddress0/1</tspan> to disk</text>

  <!-- 3. Workers self: DHT peer discovery -->
  <path d="M380 185 L 430 185 L 430 205 L 386 205" fill="none" stroke="#666" stroke-width="1.5" marker-end="url(#ipc-arr-gray)"/>
  <text x="440" y="198" font-family="Inter,sans-serif" font-size="12" fill="#444">DHT peer discovery</text>

  <!-- 4. Workers self: AllReduce within stage -->
  <path d="M380 235 L 430 235 L 430 255 L 386 255" fill="none" stroke="#666" stroke-width="1.5" marker-end="url(#ipc-arr-gray)"/>
  <text x="440" y="248" font-family="Inter,sans-serif" font-size="12" fill="#444"><tspan font-family="ui-monospace,Menlo,monospace" font-size="11">AllReduce</tspan> within stage (libp2p)</text>

  <!-- 5. Seeds → Trainers: read multiaddr file -->
  <line x1="120" y1="290" x2="638" y2="290" stroke="#1a7f37" stroke-width="1.5" stroke-dasharray="6 4" marker-end="url(#ipc-arr-disc)"/>
  <text x="380" y="282" text-anchor="middle" font-family="Inter,sans-serif" font-size="12" font-weight="600" fill="#1a7f37">read multiaddr file (via disk)</text>

  <!-- 6. Trainers → Workers: forward + backward calls (blue) -->
  <line x1="640" y1="345" x2="382" y2="345" stroke="#0969da" stroke-width="1.5" marker-end="url(#ipc-arr-blue)"/>
  <text x="511" y="337" text-anchor="middle" font-family="Inter,sans-serif" font-size="12" font-weight="600" fill="#0969da">batches &amp; activations (libp2p gRPC)</text>

  <!-- 7. Workers → Trainers: forward outputs + backward grad_input (yellow) -->
  <line x1="380" y1="395" x2="638" y2="395" stroke="#bf8700" stroke-width="1.5" marker-end="url(#ipc-arr-warn)"/>
  <text x="509" y="387" text-anchor="middle" font-family="Inter,sans-serif" font-size="12" font-weight="600" fill="#bf8700">activations / loss / grad_input</text>
</svg>
</div>
<figcaption>UML sequence diagram — time flows top to bottom. <span class="fig-discovery">Bootstrap</span> (green, dashed) is file-mediated via disk; runtime messages between Trainers and Workers are direct libp2p gRPC — <span class="fig-forward">batches &amp; activations in</span>, <span class="fig-backward">activations / loss / grad_input out</span>. Parameter gradients are computed on the workers and never cross the RPC boundary — the optimizer step happens locally. Self-loop arrows on the Workers lifeline are DHT discovery and within-stage <code>AllReduce</code>.</figcaption>
</figure>

With auth disabled, the only processes are seeds, workers, and trainers, communicating via DHT for peer discovery and libp2p for data transport. All processes run inside a single Docker container and communicate via localhost networking (`--network host`).
