"""DataFly contact-rich study (r7) — MuJoCo, Kaggle GPU.

The kinematic PlanarPusher is replaced by real contact dynamics (MuJoCo):
a 2-link arm pushes a frictional box via a fingertip sphere, so the block can
slip, spin, or stall. The observation semantics are identical, so the whole
flywheel stack (expert, policies, strategies, loop) runs unchanged — the
only differences are env_cls=MuJoCoPusher, expert_cls=PushCommitExpert, and
expert_cap_radius=0.07 (MuJoCo's sphere+box contact range).

This kernel re-measures the mechanism's two pillars on contact-rich physics:

  STUDY A — state phase diagram (contact-rich):
      capacity hidden in {32, 96}
        x strategies: relabel, relabel_curated, relabel_adaptive
        + relabel_mix at mix_ratio in {0.25, 0.5, 1.0}
      2 seeds x 4 iterations, track_train_loss=True.

  STUDY B — vision (contact-rich + pixels): relabel vs relabel_mix@0.25,
      the crash-vs-rescue comparison on contact physics with a CNN.

Results -> /kaggle/working/results_contact/{state,vision}/
"""

import glob
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
t0 = time.time()
OUT = Path("/kaggle/working")

# The mujoco wheel ships in the dataset (Kaggle internet egress is flaky and
# killed r7 — DNS resolution failed mid-pip). Install from the local wheel.
pkg = Path("/kaggle/working/pkg")
for tar in glob.glob("/kaggle/input/**/*.tar", recursive=True):
    print("extracting", tar, flush=True)
    with tarfile.open(tar) as t:
        t.extractall(pkg)
whls = sorted(glob.glob(str(pkg / "**" / "wheels" / "*.whl"), recursive=True) or
              glob.glob("/kaggle/input/**/wheels/*.whl", recursive=True))
if whls:
    # --no-deps: the image has numpy; glfw is only needed for the interactive
    # viewer, not for import or the EGL renderer. Install each wheel we ship.
    for w in whls:
        print("installing", w, flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "--no-deps", w], check=True)
else:
    print("installing mujoco from PyPI ...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mujoco"],
                   check=True)

cands = glob.glob(str(pkg / "**" / "datafly" / "__init__.py"), recursive=True) or \
    glob.glob("/kaggle/input/**/datafly/__init__.py", recursive=True)
assert cands, "datafly package not mounted"
sys.path.insert(0, os.path.dirname(os.path.dirname(cands[0])))

from datafly import FlywheelConfig, run_flywheel  # noqa: E402
from datafly.envs.mujoco_pusher import MuJoCoPusher  # noqa: E402
from datafly.policies.expert import PushCommitExpert  # noqa: E402

REFERENCE_STRATEGIES = ("relabel", "relabel_curated", "relabel_adaptive")
RATIOS = (0.25, 0.5, 1.0)
CAPACITIES = (32, 96)


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


def base_cfg(cap: int, **kw) -> FlywheelConfig:
    return FlywheelConfig(
        env_cls=MuJoCoPusher, expert_cls=PushCommitExpert, expert_cap_radius=0.07,
        hidden=cap, track_train_loss=True, verbose=False, **kw)


# --------------------------------------------------------------------- #
# STUDY A: contact-rich state phase diagram                             #
# --------------------------------------------------------------------- #
print("=== A: CONTACT-RICH STATE PHASE DIAGRAM ===", flush=True)
out = OUT / "results_contact" / "state"
out.mkdir(parents=True, exist_ok=True)
manifest = {"obs": "state", "env": "mujoco", "capacities": list(CAPACITIES),
            "ratios": list(RATIOS), "seeds": 2, "iterations": 4, "cells": {}}

for cap in CAPACITIES:
    base = base_cfg(cap, seed_demos=40, collect_per_iter=30, eval_starts=60,
                    iterations=4, seeds=2, epochs=120, finetune_epochs=40)
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
print(f"contact state phase diagram done in {time.time() - t0:.0f}s", flush=True)

# --------------------------------------------------------------------- #
# STUDY B: contact-rich vision (crash vs rescue on pixels + physics)    #
# --------------------------------------------------------------------- #
print("=== B: CONTACT-RICH VISION ===", flush=True)
try:
    vout = OUT / "results_contact" / "vision"
    vout.mkdir(parents=True, exist_ok=True)
    vman = {"obs": "image", "env": "mujoco", "seeds": 1, "iterations": 3, "cells": {}}
    vbase = base_cfg(48, obs_mode="image", img_size=48, seed_demos=24,
                     collect_per_iter=18, eval_starts=40, iterations=3, seeds=1,
                     epochs=40, finetune_epochs=15)
    for name in ("relabel", "relabel_mix"):
        kw = {"strategies": (name,)}
        if name == "relabel_mix":
            kw["strategy_kwargs"] = {"relabel_mix": {"mix_ratio": 0.25}}
        cfg = FlywheelConfig(**{**vars(vbase), "out_dir": str(vout), **kw})
        run_cell(cfg, name, vout, vman["cells"], f"vision/{name}")
    (vout / "manifest.json").write_text(json.dumps(vman, indent=2))
except Exception as e:  # noqa: BLE001 — headless GL can be flaky; the state
    print(f"vision study failed (non-fatal): {e}", flush=True)  # study is the deliverable

print(f"ALL DONE in {time.time() - t0:.0f}s — results in {OUT / 'results_contact'}", flush=True)
