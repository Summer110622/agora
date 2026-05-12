---
order: 4
---

# Protocol Overview

The system has the following roles that enable training a network:

- **Worker** — Represents a stage (composed of one or many layers). Implements the following core functionalities: forward pass, backward pass, local optimizer, all-reduce coordinator with peers in the same stage.
- **Trainer** — Sends batches of data to the workers. Has entire knowledge of network stages and orchestrates sending activations and gradients between workers of different stages. Doesn't do any actual computations.

## Additional Roles

The system as a whole has these additional roles:

- **Seed** — These nodes are launched before workers and trainers. They are used as gateway information for workers and trainers to first enter the DHT and discover further nodes.
- **Authorizer** — Authorizes a joining node (Trainer/Worker) by verifying a provided Hugging Face token, checking the source code is the latest version, and running hardware checks (GPU memory, system RAM, bandwidth and latency). Nodes re-verify periodically (every 1 hour). In case a node cannot be authorized or the source code is not the latest version, it will be removed from the run. The Authorizer also manages a joining queue, via redis backend, ensuring a joining worker downloads the stage parameters (weights and optimizer states) and assigns workers to the least-utilized body stage.

## Architecture

![Pluralis Swarm Architecture](system_architecture.svg)
