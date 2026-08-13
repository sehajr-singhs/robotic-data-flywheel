# Robotic Data Flywheel (DataFly)

> **The loop only spins if the labels come back.** Self-generated data
> plateaus; relabeling deployment failures compounds. A controlled,
> reproducible study of *curation strategy* — the one variable that decides
> whether a robot data flywheel spins up or stalls.

A data flywheel is the closed loop in which a deployed policy's rollouts are
collected, curated, and fed back into the next training run. Every serious
scaling bet in robot learning — RT-2, Open X-Embodiment, DROID, industrial
deployment programs — assumes this loop compounds. **DataFly** is the
smallest faithful implementation of that loop, built so the curation
strategy is the *only* thing that varies:

```
        ┌─────────────────────────────────────────────────────┐
        │                  THE FLYWHEEL LOOP                   │
        └─────────────────────────────────────────────────────┘
   deploy ──► collect rollouts ──► score (success / progress / ──┐
     ▲                              smoothness / coverage)      │
     │                                                           ▼
   evaluate on                   curate (the strategy decides ──┤
   held-out starts                what goes back in)             │
     ▲                                                           │
     └────────────── retrain (fine-tune from previous) ◄─────────┘
```

**Paper:** [`paper/manuscript.pdf`](paper/manuscript.pdf) (IEEE format,
compiled from `paper/manuscript.tex`). Every number in the paper reads from
committed JSON under `results/` via `scripts/render_results.py` — never
hand-typed.

## The core result

Held-out push success (mean over 4 training seeds, 200 held-out starts per
evaluation) as the flywheel turns:

| strategy | iter 0 | iter 5 | Δ |
|---|---|---|---|
| none (frozen) | 0.17 | 0.17 | — |
| self: successes | 0.20 | 0.38 | +0.18 |
| self: near-misses | 0.17 | 0.33 | +0.16 |
| self: novel successes | 0.17 | 0.38 | +0.21 |
| **oracle: curated relabel** | **0.17** | **0.43** | **+0.26** |
| **oracle: relabel all (DAgger)** | **0.17** | **0.60** | **+0.43** |

Three findings:

1. **No feedback ⇒ flat.** The frozen control sits at 0.17 — the null
   hypothesis the flywheel must beat.
2. **Self-curation ⇒ modest, plateauing gains.** The policy's own
   successful episodes densify what it already does; they cannot repair
   what it does badly.
3. **Relabeling deployment failures ⇒ the flywheel compounds.** Oracle
   relabeling (DAgger-style) more than triples success
   (0.17 → 0.60).

### Ablations (see `results/ablations.json`, `scripts/analyze.py`)

- **Label efficiency.** Curated relabeling reaches 0.43 with **7,490**
  oracle queries; blind relabeling needs **16,773** to reach 0.60 —
  curation buys ~1.6× the performance per query, and wins at any matched
  budget below the blind-relabel ceiling.
- **Oracle quality.** With a noisy relabeling oracle (human-label
  mistakes), blind relabeling loses 20% of its clean-oracle performance;
  curated relabeling loses only 13% — curation dampens label noise.
- **Difficulty.** On a harder task (tighter goal, longer pushes), the
  flywheel still compounds: relabeling 0.09 → 0.26 while no feedback stays
  flat at 0.09.

The mechanism is compounding error: BC policies drift off the expert
trajectory, and no amount of expert data fixes states the policy has never
reached. On-policy rollouts reach exactly those states — the question is
what the loop does with them.

## Repository layout

```
src/datafly/
  envs/planar_pusher.py      # 2-link arm pushing a block (numpy, fast)
  policies/expert.py         # scripted oracle (state-complete, "human labeler")
  policies/mlp.py            # BC policy (numpy MLP, manual backprop) + rollouts
  curation/scores.py         # success / progress / smoothness / coverage
  curation/strategies.py     # six ingestion rules
  loop.py                    # the flywheel driver
  eval.py                    # fixed held-out evaluation
  viz.py                     # success curves, flywheel report, trajectories
scripts/
  run_experiment.py          # run the study (per-strategy JSON)
  merge_results.py           # assemble summary.json + figures
  render_results.py          # paper tables from JSON
paper/                       # manuscript.tex (IEEEtran) + refs.bib
results/                     # committed results + figures (single source of truth)
tests/                       # 13 tests: env, expert, curation, loop determinism
```

The policy is a deliberately small numpy MLP: the point of the study is
that **data, not model scale, drives improvement**. Swap in a
vision-language-action model and the loop machinery is unchanged.

## Quickstart

```bash
# numpy-only core (torch optional, for swapping in bigger policies)
pip install -r requirements.txt

# full study (6 strategies × 5 iterations × 4 seeds, ~30 min on one CPU core)
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/run_experiment.py
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/merge_results.py --out-dir results/main
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/analyze.py

# paper tables/numbers from results JSON (never hand-typed)
PYTHONPATH=src python scripts/render_results.py

# tests
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
```

> Note: on Windows, set `OPENBLAS_NUM_THREADS=1` before importing numpy —
> single-threaded BLAS is ~100× faster for this workload's tiny matmuls, and
> keeps results bit-for-bit reproducible.

Reproduce the committed results exactly (the heavy `relabel` strategies
are run one per invocation so each stays under a 10-minute budget):

```bash
# main comparison (6 strategies × 4 seeds × 5 iterations)
for s in "none success_only" "near_miss" "relabel" "relabel_curated" "success_coverage"; do
  PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/run_experiment.py \
    --strategies $s --out-dir results/main --seed-demos 80 --collect-per-iter 60 \
    --eval-starts 200 --iterations 5 --seeds 4 --epochs 150 --finetune-epochs 50
done
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/merge_results.py --out-dir results/main

# ablations
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/analyze.py          # figures + ablations.json
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python scripts/render_results.py    # paper tables/numbers
```

## Curation strategies

| strategy | rule | class |
|---|---|---|
| `none` | no feedback, policy frozen | control |
| `success_only` | keep successful episodes | self-curation |
| `near_miss` | keep failed episodes within 0.15 of goal | self-curation |
| `success_coverage` | keep most *novel* successes (state-space coverage) | self-curation |
| `relabel` | relabel *everything* with the oracle (DAgger) | oracle |
| `relabel_curated` | relabel only failures whose progress signal says they came close | oracle |

## The flywheel report

Every iteration writes a deployment report (`results/report.json` +
`results/figs/flywheel_report.png`): outcome buckets, final-distance
histogram, and the coverage-vs-outcome scatter that drives
novelty-aware curation. This is the observability layer industrial loops
need — a loop you cannot audit is a loop you cannot trust.

## Extending

- **New task:** implement an environment with the same interface
  (`reset`, `step`, `trajectory`, `sample_start`); keep the oracle
  state-complete if you want relabel strategies.
- **Bigger policy:** implement `act(state) -> action` and pass it to
  `train_bc`'s callers; the loop is policy-agnostic by construction.
- **Noisy oracle:** the oracle is a scripted controller; add action noise to
  model human labeling mistakes and re-measure the curation tradeoff.

## Why this matters for industrial robotics

Manufacturing deployment faces exactly this trade: teleoperation hours are
scarce, rollouts are cheap, and the marginal value of a labeled episode
depends on the curation rule. The highest-leverage investment is not more
model capacity but (i) an oracle channel for labeling deployment failures
and (ii) a scoring layer that decides which failures deserve the label.

## License

MIT — see [LICENSE](LICENSE).
