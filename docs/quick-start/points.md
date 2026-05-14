---
icon: material/trophy-outline
---

# Points & Leaderboard

How your contribution to the Pluralis-8B run is measured, scored, and ranked.

---

## The short version

- Every node you run accrues a **score** based on the FLOPs it performs.
- Scores are summed across all your peers — one user, many GPUs = additive.
- Live leaderboard on the [Dashboard](https://dashboard.pluralis.ai/), refreshed every 120 s.

---

## How the score is calculated

The underlying signal is **FLOPs your peers have performed**. Each peer continuously reports two metrics:

- `total_flop` — FLOPs from completed forward / backward passes
- `running_flop` — FLOPs currently in-flight

The health-monitor service sums those across every peer registered to your user identity. Accrual is continuous — there's no per-round commit step, so no work is "lost" if you disconnect mid-batch (only the small amount of in-flight `running_flop` from that exact moment).

!!! info "What you see on the leaderboard"
    The dashboard rescales the raw FLOP total into a **normalized 0–100 score**, so the top contributor sits near 100 and everyone else is ranked relative to them. Higher score = more FLOPs contributed.

---

## What the leaderboard shows

| Column          | What it means                                                                        |
| --------------- | ------------------------------------------------------------------------------------ |
| **Rank**        | Position on `score`, descending. `1` is the top contributor.                         |
| **Username**    | The ID we sent you to your logds.                                                    |
| **Active time** | Cumulative **hours** your peer machines have been online since you joined.           |
| **Joined**      | Unix timestamp of when your account first appeared.                                  |
| **Score**       | Normalized contribution score, `0` – `100`. Higher = more FLOPs.                     |
| **Status**      | `online` or `offline`. Blacklisted accounts are excluded from the response entirely. |

The dashboard re-queries the live leaderboard every 120 seconds, so a fresh contribution shows up on the public board within a couple of minutes.

---

## How to maximize your score

- **Better hardware** — more FLOPs per second translates directly to a higher score per minute.
- **Stable network** — disconnects don't lose accrued FLOPs, but they pause accrual until the node rejoins.
- **Long uptime** — score accrues continuously; running 24/7 beats running 4 hr/day at the same hardware.
- **Multi-GPU** — each GPU runs as a separate peer; their scores sum under your user identity.

---

## Eligibility

- Volunteer participants only. Peers run on Pluralis-owned infrastructure (Seeds, Trainers, internal Workers) are excluded from the public leaderboard.
- `last_active` must not be null — peers that have never reported in won't appear.
- Blacklisted accounts (rule violations) are excluded entirely.
