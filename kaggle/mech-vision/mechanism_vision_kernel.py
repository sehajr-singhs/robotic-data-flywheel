"""DataFly mechanism study, part 2 (vision) — Kaggle GPU.

The crash lives in perception space: the r4/r5 runs showed blind relabeling
collapses the CNN (0.27 -> 0.06) and even ratio-capped-at-1.0 relabeling
still floods (the cap scaled with the growing dataset). This kernel maps the
CNN's stability region with the *true* controlled variable — the
relabeled:clean mix ratio held fixed in the training set (`relabel_mix`) —
plus the closed-loop curator (`relabel_adaptive`):

  capacity hidden in {32, 128}
    x strategies: relabel, relabel_curated, relabel_adaptive
    + relabel_mix at mix_ratio in {0.25, 0.5, 1}
  2 seeds x 4 iterations, 64x64 RGB, track_train_loss=True.

The phase boundary (smallest ratio that still collapses) is the paper's
mechanism result; the adaptive controller should converge to a ratio near it
without knowing capacity.

Results -> /kaggle/working/results_mech/vision/
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

REFERENCE_STRATEGIES = ("relabel", "relabel_curated", "relabel_adaptive")
RATIOS = (0.25, 0.5, 1.0)
CAPACITIES = (32, 128)


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


print("=== VISION PHASE DIAGRAM ===", flush=True)
out = OUT / "results_mech" / "vision"
out.mkdir(parents=True, exist_ok=True)
manifest = {"obs": "image", "capacities": list(CAPACITIES), "ratios": list(RATIOS),
            "seeds": 2, "iterations": 4, "cells": {}}

for cap in CAPACITIES:
    base = FlywheelConfig(
        obs_mode="image", img_size=64, seed_demos=40, collect_per_iter=30,
        eval_starts=120, iterations=4, seeds=2, hidden=cap, epochs=60,
        finetune_epochs=20, track_train_loss=True, verbose=False,
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
print(f"vision phase diagram done in {time.time() - t0:.0f}s", flush=True)
print(f"ALL DONE in {time.time() - t0:.0f}s", flush=True)
