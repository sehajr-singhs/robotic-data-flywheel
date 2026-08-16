"""DataFly mechanism study, part 1 (state space + DQN tuning) — Kaggle GPU.

The flood hypothesis: blind relabeling grows the dataset's relabeled:clean
ratio without bound and the policy overfits its own failure distribution.
This kernel maps the stability region in (capacity x mix-ratio) space for the
state MLP and tunes the DQN anchor so the RL baseline is competitive.

  STUDY A — phase diagram (state):
      capacity hidden in {32, 96, 256}
        x strategies: relabel, relabel_curated, relabel_adaptive
        + relabel_mix at mix_ratio in {0.1, 0.25, 0.5, 1, 2}
      3 seeds x 6 iterations, track_train_loss=True (the overfitting proxy).

  STUDY B — DQN tuning:
      8 hyperparameter configs x 2 seeds x 300k env steps, double-Q,
      success-vs-steps curve for the best config (the honest RL anchor).

Results -> /kaggle/working/results_mech/state/ and /kaggle/working/results_dqn/
"""

import glob
import json
import os
import sys
import tarfile
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
t0 = time.time()
OUT = Path("/kaggle/working")

cands = glob.glob("/kaggle/input/**/datafly/__init__.py", recursive=True)
if not cands:  # dataset shipped as tarball(s): extract then import
    pkg = Path("/kaggle/working/pkg")
    for tar in glob.glob("/kaggle/input/**/*.tar", recursive=True):
        print("extracting", tar, flush=True)
        with tarfile.open(tar) as t:
            t.extractall(pkg)
    cands = glob.glob(str(pkg / "**" / "datafly" / "__init__.py"), recursive=True)
assert cands, "datafly package not mounted"
sys.path.insert(0, os.path.dirname(os.path.dirname(cands[0])))

from datafly import FlywheelConfig, run_flywheel  # noqa: E402
from datafly.envs.planar_pusher import PlanarPusher  # noqa: E402
from datafly.eval import make_eval_starts  # noqa: E402
from datafly.policies.dqn import train_dqn  # noqa: E402

REFERENCE_STRATEGIES = ("relabel", "relabel_curated", "relabel_adaptive")
RATIOS = (0.1, 0.25, 0.5, 1.0, 2.0)
CAPACITIES = (32, 96, 256)


def run_cell(cfg, name, out_dir, manifest, label):
    print(f"\n=== {label}: {name} ===", flush=True)
    s = run_flywheel(cfg)
    cell = out_dir / name
    cell.mkdir(parents=True, exist_ok=True)
    for sname, data in s["strategies"].items():
        (cell / f"strategy_{sname}.json").write_text(json.dumps(data, indent=2))
    (cell / "run_config.json").write_text(json.dumps({"config": s["config"],
                                                      "cell": label}, indent=2))
    manifest[label] = {"strategies": {sname: {
        "success_rate_mean": data["success_rate_mean"],
        "curation_log": data.get("curation_log")} for sname, data in s["strategies"].items()},
        "config": s["config"]}


# --------------------------------------------------------------------- #
# STUDY A: state phase diagram                                          #
# --------------------------------------------------------------------- #
print("=== A: STATE PHASE DIAGRAM ===", flush=True)
out = OUT / "results_mech" / "state"
out.mkdir(parents=True, exist_ok=True)
manifest = {"obs": "state", "capacities": list(CAPACITIES), "ratios": list(RATIOS),
            "seeds": 3, "iterations": 6, "cells": {}}

for cap in CAPACITIES:
    base = FlywheelConfig(
        seed_demos=100, collect_per_iter=80, eval_starts=300,
        iterations=6, seeds=3, hidden=cap, epochs=150, finetune_epochs=50,
        track_train_loss=True, verbose=False,
    )
    for name in REFERENCE_STRATEGIES:
        cfg = FlywheelConfig(**{**vars(base), "strategies": (name,),
                                "out_dir": str(out)})
        run_cell(cfg, name, out, manifest["cells"], f"cap{cap}/{name}")
    for r in RATIOS:
        cfg = FlywheelConfig(**{**vars(base), "strategies": ("relabel_mix",),
                                "strategy_kwargs": {"relabel_mix": {"mix_ratio": r}},
                                "out_dir": str(out)})
        run_cell(cfg, f"r{r:g}", out, manifest["cells"], f"cap{cap}/mix{r:g}")

(out / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"state phase diagram done in {time.time() - t0:.0f}s", flush=True)

# --------------------------------------------------------------------- #
# STUDY B: DQN tuning                                                   #
# --------------------------------------------------------------------- #
print("=== B: DQN TUNING ===", flush=True)
import numpy as np  # noqa: E402
from itertools import product  # noqa: E402

env = PlanarPusher(seed=0)
starts = make_eval_starts(env, 200, seed=2)
dout = OUT / "results_dqn"
dout.mkdir(parents=True, exist_ok=True)
results = {"configs": {}, "curves": {}}
best = None
for lr, hidden, target_every in product((3e-4, 1e-3), (128, 256), (1_000, 2_000)):
    key = f"lr{lr:g}_h{hidden}_te{target_every}"
    seeds_out = []
    for seed in range(2):
        r = train_dqn(env, starts, budget=300_000, seed=seed, hidden=hidden,
                      lr=lr, target_every=target_every, eval_every=25_000,
                      verbose=True)
        seeds_out.append(r)
        (dout / f"{key}_seed{seed}.json").write_text(json.dumps(r, indent=2))
    finals = [s["final_success"] for s in seeds_out]
    mean_final = float(np.mean(finals))
    results["configs"][key] = {"lr": lr, "hidden": hidden,
                               "target_every": target_every,
                               "final_success_mean": mean_final,
                               "final_success_seeds": finals}
    results["curves"][key] = seeds_out[0]["eval"]
    if best is None or mean_final > best["final_success_mean"]:
        best = {**results["configs"][key], "key": key}
    print(f"  {key}: final {mean_final:.3f}", flush=True)

results["best"] = best
results["budget"] = 300_000
(dout / "summary.json").write_text(json.dumps(results, indent=2))
print(f"dqn tuning done in {time.time() - t0:.0f}s", flush=True)
print(f"ALL DONE in {time.time() - t0:.0f}s", flush=True)
