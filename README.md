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

Held-out push success (mean over 6 training seeds, 300 held-out starts per
evaluation, GPU study) as the flywheel turns:

| strategy | iter 0 | iter 6 | Δ |
|---|---|---|---|
| none (frozen) | 0.12 | 0.12 | — |
| self: successes | 0.15 | 0.34 | +0.19 |
| self: near-misses | 0.14 | 0.32 | +0.18 |
| self: novel successes | 0.17 | 0.37 | +0.20 |
| **oracle: curated relabel** | **0.20** | **0.48** | **+0.28** |
| **oracle: relabel all (DAgger)** | **0.15** | **0.66** | **+0.51** |

Three findings:

1. **No feedback ⇒ flat.** The frozen control sits at 0.12 — the null
   hypothesis the flywheel must beat.
2. **Self-curation ⇒ modest, plateauing gains.** The policy's own
   successful episodes densify what it already does; they cannot repair
   what it does badly.
3. **Relabeling deployment failures ⇒ the flywheel compounds.** Oracle
   relabeling (DAgger-style) more than quadruples success
   (0.15 → 0.66).

### Ablations (see `results/ablations.json`, `scripts/analyze.py`)

- **Label efficiency.** Curated relabeling reaches 0.48 with ~11k
  oracle queries; blind relabeling needs ~27k to reach 0.66 — curation
  buys ~1.4× the performance per query, and wins at any matched budget.
- **Oracle quality.** With a noisy relabeling oracle (human-label
  mistakes), blind relabeling loses 20% of its clean-oracle performance;
  curated relabeling loses only 13% — curation dampens label noise.
- **Difficulty.** On a harder task (tighter goal, longer pushes), the
  flywheel still compounds: relabeling 0.09 → 0.26 while no feedback stays
  flat.

### v3 scale-up: perception + an RL anchor (GPU, `kaggle/`)

- **The loop transfers to raw pixels.** A torch CNN on
  64×64 RGB renders behaves like the state MLP under *curated*
  relabeling — but **blind relabeling crashes it** (0.27 → 0.06): the
  relabeled frames flood the dataset (14× the clean demos) and the
  high-capacity CNN overfits its own failure distribution. Curation is
  what keeps the loop stable in perception space too.
- **Flywheel ≫ RL from scratch.** A DQN trained from scratch with a dense
  reward reaches 0.045 success after 300,000 environment interactions;
  the flywheel reaches 0.66 with ~29k collected frames — a ~10×
  interaction budget at >14× the success rate. This is the
  label-efficiency argument in its strongest form: the loop converts
  *existing* deployment telemetry into signal; it does not pay the
  exploration cost.

The mechanism is compounding error: BC policies drift off the expert
trajectory, and no amount of expert data fixes states the policy has never
reached. On-policy rollouts reach exactly those states — the question is
what the loop does with them.

## Repository layout

```
src/datafly/
  envs/planar_pusher.py      # 2-link arm pushing a block (numpy, fast) + pixel renderer
  policies/expert.py         # scripted oracle (state-complete, "human labeler")
  policies/mlp.py            # BC policy (numpy MLP, manual backprop) + rollouts
  policies/cnn.py            # torch CNN vision policy (BC, optional dep)
  policies/dqn.py            # DQN-from-scratch RL baseline (optional dep)
  curation/scores.py         # success / progress / smoothness / coverage
  curation/strategies.py     # seven ingestion rules (incl. relabel_balanced)
  loop.py                    # the flywheel driver (obs_mode: state|image)
  eval.py                    # fixed held-out evaluation
  viz.py                     # success curves, flywheel report, trajectories
scripts/
  run_experiment.py          # run the study (per-strategy JSON)
  merge_results.py           # assemble summary.json + figures
  rebuild_summary.py         # reassemble summary.json from kernel per-strategy files
  analyze.py                 # ablation figures + budget comparison
  render_results.py          # paper tables from JSON
  build_site.py              # GitHub Pages site
kaggle/                      # GPU kernels that produced the v3 results
paper/                       # manuscript.tex (IEEEtran) + refs.bib
results/                     # committed results + figures (single source of truth)
tests/                       # 17 tests: env, expert, curation, loop, vision
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
