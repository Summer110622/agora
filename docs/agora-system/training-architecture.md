---
icon: octicons/cpu-16
---

# Training Architecture { .numbered }

Pipeline-parallel training with DHT-based peer discovery, heterogeneous workers, and CPU-only trainers.

The system is organized into three layers — **Discovery**, **Compute**, and **Coordination** — each owning a distinct responsibility. Together they enable distributed training across heterogeneous, untrusted hardware while ensuring no single participant holds complete model weights.

<figure class="agora-figure" markdown="0">
<div class="swarm-fig">
  <div class="architecture">
    <div class="zone seed-zone">
      <div class="zone-header">
        <div class="zone-number">Discovery Layer</div>
        <div class="zone-title">Seeds</div>
        <div class="zone-desc">DHT bootstrap nodes (Kademlia). Peer registration &amp; discovery.</div>
      </div>
      <div class="zone-body">
        <div class="seed-item">
          <div class="seed-name">Seed 0</div>
          <div class="seed-desc">Primary DHT bootstrap. Publishes multiaddr for initial peer connections.</div>
        </div>
        <div class="seed-item">
          <div class="seed-name">Seed 1</div>
          <div class="seed-desc">Redundant bootstrap. Ensures discovery continues if Seed 0 is unreachable.</div>
        </div>
      </div>
      <div class="zone-footer">
        <div class="security-badge">
          <div class="badge-dot"></div>
          <div class="badge-text"><strong>Stateless.</strong> Routing only, no model data.</div>
        </div>
      </div>
    </div>
    <div class="arrow-col">
      <div class="data-flow">
        <div class="flow-direction">Peer Discovery</div>
        <div class="flow-arrow-line flow-discovery"><div class="line"></div><div class="head-right"></div></div>
        <div class="flow-payload">
          <div class="payload-what">Multiaddr exchange</div>
          <div class="payload-size">libp2p / Kademlia</div>
        </div>
      </div>
    </div>
    <div class="zone pipeline-zone">
      <div class="zone-header">
        <div class="zone-number">Compute Layer</div>
        <div class="zone-title">Training Pipeline</div>
        <div class="zone-desc">Workers hold model shards, process fwd/bwd, run periodic SPARTA state averaging within stage.</div>
      </div>
      <div class="zone-body">
        <div class="stage-grid">
          <div class="pipeline-stage head">
            <div class="stage-label">Stage 0</div>
            <div class="stage-name">Head</div>
            <div class="worker-node"><div class="worker-label">Pipe 0</div><div class="worker-gpu">H100 80G</div><div class="worker-info">Layers 0&ndash;7, embed</div></div>
            <div class="worker-node"><div class="worker-label">Pipe 1</div><div class="worker-gpu">H100 80G</div><div class="worker-info">Layers 0&ndash;7, embed</div></div>
            <div class="ar-bar"><div class="ar-label">SPARTA</div><div class="ar-detail">State averaging</div></div>
          </div>
          <div class="pipeline-stage">
            <div class="stage-label">Stage 1</div>
            <div class="stage-name">Body 1</div>
            <div class="worker-node"><div class="worker-label">Pipe 0</div><div class="worker-gpu">RTX 4090</div><div class="worker-info">Layers 8&ndash;15</div></div>
            <div class="worker-node"><div class="worker-label">Pipe 1</div><div class="worker-gpu">RTX 4090</div><div class="worker-info">Layers 8&ndash;15</div></div>
            <div class="ar-bar"><div class="ar-label">SPARTA</div><div class="ar-detail">State averaging</div></div>
          </div>
          <div class="pipeline-stage">
            <div class="stage-label">Stage 2</div>
            <div class="stage-name">Body 2</div>
            <div class="worker-node"><div class="worker-label">Pipe 0</div><div class="worker-gpu">L40S 48G</div><div class="worker-info">Layers 16&ndash;23</div></div>
            <div class="worker-node"><div class="worker-label">Pipe 1</div><div class="worker-gpu">L40S 48G</div><div class="worker-info">Layers 16&ndash;23</div></div>
            <div class="ar-bar"><div class="ar-label">SPARTA</div><div class="ar-detail">State averaging</div></div>
          </div>
          <div class="pipeline-stage tail">
            <div class="stage-label">Stage 3</div>
            <div class="stage-name">Tail</div>
            <div class="worker-node"><div class="worker-label">Pipe 0</div><div class="worker-gpu">A100 40G</div><div class="worker-info">Layers 24&ndash;31, lm_head</div></div>
            <div class="worker-node"><div class="worker-label">Pipe 1</div><div class="worker-gpu">A100 40G</div><div class="worker-info">Layers 24&ndash;31, lm_head</div></div>
            <div class="ar-bar"><div class="ar-label">SPARTA</div><div class="ar-detail">State averaging</div></div>
          </div>
        </div>
      </div>
      <div class="zone-footer">
        <div class="security-badge">
          <div class="badge-dot"></div>
          <div class="badge-text"><strong>No party holds complete weights</strong> &mdash; mathematical guarantee.</div>
        </div>
      </div>
    </div>
    <div class="arrow-col">
      <div class="data-flow">
        <div class="flow-direction">Activations &rarr;</div>
        <div class="flow-arrow-line flow-forward"><div class="line"></div><div class="head-right"></div></div>
        <div class="flow-payload"><div class="payload-what">Forward pass</div><div class="payload-size">Head &rarr; Body &rarr; Tail</div></div>
      </div>
      <div class="data-flow">
        <div class="flow-direction">&larr; Gradients</div>
        <div class="flow-arrow-line flow-backward"><div class="line"></div><div class="head-left"></div></div>
        <div class="flow-payload"><div class="payload-what">Backward pass</div><div class="payload-size">Tail &rarr; Body &rarr; Head</div></div>
      </div>
    </div>
    <div class="zone trainer-zone">
      <div class="zone-header">
        <div class="zone-number">Coordination Layer</div>
        <div class="zone-title">Trainers</div>
        <div class="zone-desc">Microbatch coordination, load balancing, data management.</div>
      </div>
      <div class="zone-body">
        <div class="trainer-item">
          <div class="trainer-name">Trainer 0</div>
          <div class="trainer-desc">Microbatch coordination<br>Load balancing<br>Data management</div>
          <div class="trainer-tag tag-cpu">CPU only</div>
        </div>
        <div class="trainer-item">
          <div class="trainer-name">Trainer 1</div>
          <div class="trainer-desc">Microbatch coordination<br>Load balancing<br>Data management</div>
          <div class="trainer-tag tag-cpu">CPU only</div>
        </div>
      </div>
      <div class="zone-footer">
        <div class="security-badge">
          <div class="badge-dot"></div>
          <div class="badge-text"><strong>Pluralis-owned only</strong> &mdash; not on volunteer nodes.</div>
        </div>
      </div>
    </div>
  </div>
</div>
<figcaption>Three-zone Agora architecture — Discovery (Seeds), Compute (Workers), Coordination (Trainers). Heterogeneous example configuration; real swarms vary by participant hardware. <span class="fig-forward">Forward</span> activations flow Head → Body → Tail; <span class="fig-backward">backward</span> gradients flow Tail → Body → Head.</figcaption>
</figure>

!!! info "*Epoch* terminology"
    In Agora, an **epoch** is a per-stage counter: every time the workers in a stage have collectively processed a number of batches equal to the target batch size, every worker in that stage increments its epoch counter by one. This is unrelated to a full pass over the dataset.

## Discovery Layer — Seeds { .t-discovery }

Seed nodes are stateless <span class="t-discovery">DHT</span> bootstrap peers — they help workers find each other but never store model data.

## Workers { .t-compute }

Workers hold the actual model parameters. Each worker owns one pipeline stage (Head / Body / Tail), runs forward and backward passes, and participates in periodic SPARTA state averaging within its stage. Two pipes per stage give data-parallel redundancy and let SPARTA averaging run with at least two contributors. See the [Workers deep dive](#workers_1) for runtime structure.

## Coordination Layer — Trainers { .t-coord }

Trainers hold no model parameters. They route microbatches through the pipeline, balance load across workers, and feed dataset shards into the run. They run on CPU and are kept on Pluralis-owned infrastructure — never deployed on volunteer nodes. See the [Trainers deep dive](#trainers) for fault tolerance and load-balancing logic.

---

## Component Deep Dive

### Workers

Workers are the GPU-bound compute nodes of the system. Each worker holds the parameters for exactly one pipeline stage (head, body, or tail) and is responsible for running forward passes, backward passes, and participating in AllReduce for that stage.

#### Architecture

Each worker wraps a single **expert** — a sub-network corresponding to one pipeline stage. The model is split so that the head stage handles input embeddings and early transformer layers, body stages handle intermediate layers, and the tail stage handles the final layers and the language model head.

The expert is wrapped in `ModuleCollab`, a subclass of Hivemind's `ModuleBackend` that adds mixed-precision (BF16 autocast) and `torch.compile` support around the standard forward/backward task pools.

Key properties of workers:

- Each worker gets a port allocated by the launch scripts (default convention: `12000 + 10 * worker_index`) and listens for gRPC requests from trainers.
- **Pipe 0** workers receive the `--seed_peer` flag and initialize weights from scratch. All other workers download state from existing peers before beginning training.
- Workers do not communicate with each other except during AllReduce rounds.
- Each worker runs on a single GPU. Multiple workers may share a GPU in local testing.

!!! info "SPARTA — Asynchronous Sparse Parameter Averaging"
    The collaborative optimizer used by workers (`OptimizerAsyncSparta`) implements the SPARTA / AsyncMesh protocol. Paper: [AsyncMesh: Fully Asynchronous Optimization for Data and Pipeline Parallelism](https://arxiv.org/pdf/2601.22442).

#### Config Loading

Workers use a three-tier YAML hierarchy, loaded bottom-up via deep recursive merge.

```text
worker_configs/default.yaml                     # root defaults (ports, compression, log rules)
  +-- worker_configs/llama/default.yaml          # model defaults (model_name, optim, averager, lr)
        +-- llama/debug_sparta_async_C.yaml      # variant overrides (num_stages, target_batch_size)
```

#### Startup Flow

The worker startup proceeds through initialization, expert construction, and optional state download from peers.

1. Parse CLI args and merge the YAML config hierarchy
2. Load model arguments from the registry (e.g., `debugmodel_C`)
3. Connect to seed nodes via <span class="t-discovery">DHT</span>
4. Generate expert UIDs and check for duplicates in the <span class="t-discovery">DHT</span>
5. Instantiate the expert model, filter parameters into decay/no-decay groups
6. Wrap the expert in `ModuleCollab` (forward/backward task pools + mixed precision)
7. Move the expert to GPU and run `torch.compile` warmup (dummy forward/backward to trigger JIT compilation)
8. Create the collaborative optimizer (`OptimizerAsyncSparta`)
9. Load weights: seed peers skip this step and use random init; non-seed peers call `load_state_from_peers()` and block until state is downloaded
10. Wait for the join window (delays joining until the swarm is not mid-AllReduce)

#### Runtime Threads

| Thread | Purpose |
|---|---|
| **Runtime** | Polls forward/backward TaskPools, processes batches, sends outputs back |
| **ConnectionHandler** | Receives gRPC requests from trainers, deserializes tensors, submits to TaskPools. Multiple `ConnectionHandler` instances run concurrently on the same port, each its own thread. |
| **DHTHandlerThread** | Periodically declares experts to the <span class="t-discovery">DHT</span> with current ghost phase (default 30 s update period) |
| **CheckpointSaver** | Periodically saves model state to disk or S3 (only when checkpointing is configured) |

#### Batch Processing

```text
Trainer sends gRPC request
  -> ConnectionHandler.rpc_forward() deserializes tensors
    -> forward_pool.submit_task(*inputs), queued
      -> TaskPool accumulates min_batch_size..max_batch_size tasks
        -> Sends batch to Runtime via pipe
          -> Runtime calls ModuleCollab.forward(*batch)
            -> Expert forward pass on GPU
          -> Output sent back through pipe
        -> TaskPool dispatches to individual Futures
      -> ConnectionHandler serializes result
    -> gRPC response back to trainer
```

The backward path is identical but calls `ModuleCollab.backward()`, which re-runs the forward pass (because pipeline-stage inputs are detached between workers — see [Memory & Communication](memory-communication.md) for why this is **not** gradient checkpointing), calls `torch.autograd.backward()`, and invokes `on_backward(batch_size)` to trigger the optimizer step.

#### DHT { .t-discovery }

Workers use Hivemind's Kademlia-based <span class="t-discovery">DHT</span> [unchanged from the original implementation](https://learning-at-home.readthedocs.io/en/latest/modules/dht.html) for peer discovery, expert registration, progress tracking, and matchmaking for SPARTA averaging rounds. Only seed nodes patch the <span class="t-discovery">DHT</span> protocol — workers run the stock library.

---

### Trainers

The trainer is a **HuggingFace Trainer** that holds **no model parameters**. It contains a dummy `nn.Parameter(randn(1,1))` so that the HuggingFace API does not raise errors, along with a `NoOpOptimizer` and `NoOpScheduler` that perform no work. The real model lives on workers — the trainer sends batches through the pipeline via remote calls.

#### Key Properties

- Configured to run on **CPU only** via `--no_cuda`. All tensors received from workers are stored in CPU RAM. (CPU-only is a launch-time convention, not enforced server-side.)
- Each trainer gets a different data shard via `--data_idx`, ensuring trainers process different portions of the dataset.
- Multiple trainers are supported and run independently. Workers serve all incoming requests regardless of which trainer sent them.
- Listens on a port allocated by the launch scripts (default convention: `49220 + 10 * trainer_idx`).

#### The Pipeline Model: TrainerLMNStage

The trainer wraps remote expert calls into a pipeline via `TrainerLMNStage`, a `torch.nn.Module`:

```python
head = BalancedRemoteExpert(dht=dhts[0], uid_prefix="head.", ...)
body = [BalancedRemoteExpert(dht=dhts[i], uid_prefix=f"body{i}.", ...) for each body stage]
tail = BalancedRemoteExpert(dht=dhts[-1], uid_prefix="tail.", ...)
```

The forward pass sends input tokens through head, then each body stage in sequence, then tail with shifted labels for next-token prediction. Each `BalancedRemoteExpert` call is a **network RPC** to a worker via libp2p, with an internal `ExpertBalancer` selecting which specific worker to call.

#### Load Balancing: ExpertBalancer

`ExpertBalancer` keeps a min-heap priority queue of workers per stage, keyed by accumulated virtual runtime. Every request is routed to the least-loaded worker; after the worker completes the request, its runtime is incremented by the estimated task duration and it re-enters the heap. New workers join at the back of the queue.

- **P2P dial failures** — the worker is short-banned for `update_period * 2` seconds (≈ 60 s with the default `update_period=30`).
- **Expert verification failures** — the worker is banned for 3600 s.
- **Refresh** — the trainer re-queries each stage's full expert list every 600 s (`refresh_experts_period`).

#### Fault Tolerance: RobustTrainer

Fault tolerance has two nested layers:

1. **Per-RPC retries** inside `BalancedRemoteExpert` — transient errors trigger a retry against the same or a different worker. Workers that keep failing are short-banned (see [Load Balancing](#load-balancing-expertbalancer)).
2. **Whole-step catch** in `RobustTrainer` — wraps every HuggingFace `training_step`. If the inner retry loop ultimately raises (worker down, timeout, etc.), `RobustTrainer` logs a warning, returns `torch.tensor(0.0)` as the loss, and continues to the next batch. A single failed batch never stops training.

If every worker in a stage is unreachable at the same time, the inner retry loop blocks until one becomes available again — at which point training resumes for that pipeline step.

#### <span class="t-discovery">DHT</span> Allocation

Workers connect to a single <span class="t-discovery">DHT</span> instance (their stage's). Trainers connect to **all** <span class="t-discovery">DHT</span> instances — one per stage in `client_mode=True` for expert discovery, plus one dedicated trainer <span class="t-discovery">DHT</span> (not in client mode) for progress reporting and commands.

Each stage's workers register themselves in the <span class="t-discovery">DHT</span> under a stage-prefixed UID — for example `head.0.0`, `body1.0.1`, `tail.0.0`. The trainer queries those prefixes to discover available workers and refreshes its full per-stage view every 600 s.
