"""DataFly v3 NMI scale-up kernel (runs on Kaggle GPU).

Three studies, all installed fresh from the public GitHub repo:

  1. MAIN   — state-based flywheel, 6 seeds x 6 iterations x 6 strategies,
              300 held-out eval starts (the headline comparison).
  2. VISION — pixel observations (64x64 RGB) -> torch CNN, the perception
              story; 4 seeds x 5 iterations x 3 strategies on GPU.
  3. DQN    — classic RL from scratch (no demonstrations), dense reward,
              300k environment steps, success-vs-interaction curve for the
              label-efficiency comparison.

Results are written to /kaggle/working/results_v3/ and pulled back via
`kaggle kernels output`.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

t0 = time.time()
OUT = Path("/kaggle/working/results_v3")

# The datafly package ships as a Kaggle dataset (always mounted, no
# internet egress needed) — see kaggle/dataset-metadata.json.
import tarfile  # noqa: E402

print("unpacking datafly from dataset ...", flush=True)
tarfile.open("/kaggle/input/datafly-v3-src/src.tar").extractall("/kaggle/working")
sys.path.insert(0, "/kaggle/working/src")

import numpy as np  # noqa: E402

from datafly import FlywheelConfig, run_flywheel  # noqa: E402
from datafly.eval import make_eval_starts  # noqa: E402
from datafly.envs.planar_pusher import PlanarPusher  # noqa: E402
from datafly.policies.dqn import train_dqn  # noqa: E402


def write_strategy_results(cfg, summary, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in summary["strategies"].items():
        (out / f"strategy_{name}.json").write_text(json.dumps(data, indent=2))
    if summary["report"]:
        (out / "report.json").write_text(json.dumps(summary["report"], indent=2))
    (out / "run_config.json").write_text(json.dumps({
        "config": summary["config"],
        "seed_expert_success_rate": summary["seed_expert_success_rate"],
        "report_strategy": cfg.report_strategy,
    }, indent=2))


# --------------------------------------------------------------------- #
# 1) main study                                                          #
# --------------------------------------------------------------------- #
print("=== MAIN STUDY (state, 6 seeds x 6 iters) ===", flush=True)
cfg = FlywheelConfig(
    seed_demos=100, collect_per_iter=80, eval_starts=300, iterations=6, seeds=6,
    strategies=("none", "success_only", "near_miss", "relabel",
                "relabel_curated", "success_coverage"),
    epochs=150, finetune_epochs=50, hidden=96,
    out_dir=str(OUT / "main"), report_strategy="relabel_curated", verbose=True,
)
s = run_flywheel(cfg)
write_strategy_results(cfg, s, OUT / "main")
print(f"main study done in {time.time() - t0:.0f}s", flush=True)

# --------------------------------------------------------------------- #
# 2) vision study                                                        #
# --------------------------------------------------------------------- #
print("=== VISION STUDY (image, torch CNN) ===", flush=True)
cfgv = FlywheelConfig(
    obs_mode="image", img_size=64, seed_demos=60, collect_per_iter=50,
    eval_starts=200, iterations=5, seeds=4,
    strategies=("none", "relabel", "relabel_curated"),
    epochs=80, finetune_epochs=30, hidden=128,
    out_dir=str(OUT / "vision"), report_strategy="relabel_curated", verbose=True,
)
sv = run_flywheel(cfgv)
write_strategy_results(cfgv, sv, OUT / "vision")
print(f"vision study done in {time.time() - t0:.0f}s", flush=True)

# --------------------------------------------------------------------- #
# 3) DQN baseline                                                        #
# --------------------------------------------------------------------- #
print("=== DQN BASELINE (300k steps, from scratch) ===", flush=True)
env = PlanarPusher(seed=0)
starts = make_eval_starts(env, 200, seed=2)
res = train_dqn(env, starts, budget=300_000, seed=0, eval_every=20_000, verbose=True)
(OUT / "dqn").mkdir(parents=True, exist_ok=True)
(OUT / "dqn" / "dqn.json").write_text(json.dumps(res, indent=2))
print(f"dqn done in {time.time() - t0:.0f}s", flush=True)

print(f"ALL DONE in {time.time() - t0:.0f}s — results in {OUT}", flush=True)
