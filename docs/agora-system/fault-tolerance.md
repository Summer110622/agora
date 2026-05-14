---
icon: material/shield-half-full
---

# Fault Tolerance & Ghost Mode

How Agora keeps training alive when peers die mid-step, and how new workers join a swarm that's already running.

---

## Worker Failure During AllReduce { .t-failure }

The `AllReduce` protocol handles peer failures gracefully at multiple levels:

* **Sender timeout** — If a peer does not send data within the configured timeout, it is **banned** (excluded from the round).
* **Mid-stream failure** — `register_failed_reducer()` marks the peer as failed and zeroes their missing contribution.
* **Outcome** — The round completes with the remaining contributors. The weighted average is computed from whoever is still alive.
* **No retry** — A failed round is logged and training continues. The next epoch's averaging round will try again with whoever is available.

If **all** peers fail, the `AllReduce` raises a `MatchmakingException`, which is caught and logged. The `finished_averaging_round` flag is set to unblock the main thread, and training continues without averaging for that round.

!!! info "Key point"
    A worker failure is fatal only if an entire pipeline stage empties. As long as at least one worker remains in each stage, the trainer can route batches and training continues.

---

## Ghost Mode Lifecycle { .t-failure }

Ghost mode manages the lifecycle of workers that join a running swarm or have fallen behind. It prevents stale or uninitialized workers from corrupting the model by gradually integrating them.

### Entry condition

A worker enters ghost mode when `global_epoch - local_epoch > max_allowed_stale`. This indicates the worker's state is too far behind the rest of the swarm to participate normally.

### Phase 1: receive-only

* The worker is **invisible to trainers** — it declares its current `ghost_phase` field on its DHT record (rather than a UID suffix), and trainers filter out workers in Phase 1.
* Does **not** process batches (no forward/backward).
* Participates in state averaging with `weight=0` — receives averaged parameters from peers but does not contribute its own.
* Duration: `ghost_phase1_steps` local epochs.

### Phase 2: warm-up

* The worker becomes **visible to trainers** and **processes real batches** for optimizer warm-up.
* Still uses `weight=0` in averaging — continues receiving without contributing.
* Duration: `ghost_phase2_steps` local epochs.

### Exit: full participation

After completing both phases, the worker returns to `GhostPhase.OFF` and participates fully — contributing its weight to averaging and counting toward epoch progress.

---

## Join Blocking Window

To prevent state corruption from workers joining mid-`AllReduce`, new peers are blocked from joining during epochs near an averaging round. This ensures that new workers only download state during a stable window when no averaging is in flight and the parameters are consistent.
