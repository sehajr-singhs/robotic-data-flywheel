"""DataFly vision study (r5): the ratio-control experiment.

The r4 vision study crashed under blind relabeling (CNN 0.27 -> 0.06): the
relabeled frames flood the dataset (3.4x the clean demos) and the
high-capacity CNN overfits its own failure distribution. This kernel tests
the ratio-control hypothesis — `relabel_balanced` caps the frames added per
iteration at a multiple of the current dataset:

  none              : frozen control
  relabel           : blind DAgger (expect the crash to reproduce)
  relabel_curated   : progress-signal curation (expect mild decline)
  relabel_balanced  : ratio-capped relabeling (the fix hypothesis)

Results -> /kaggle/working/results_v3/vision/ (overwrites r4 vision).
"""

import glob
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
t0 = time.time()
OUT = Path("/kaggle/working/results_v3")

cands = glob.glob("/kaggle/input/**/datafly/__init__.py", recursive=True)
assert cands, "datafly package not mounted"
sys.path.insert(0, os.path.dirname(os.path.dirname(cands[0])))

import numpy as np  # noqa: E402

from datafly import FlywheelConfig, run_flywheel  # noqa: E402


def write_results(cfg, summary, out_dir):
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


print("=== VISION RATIO-CONTROL STUDY (CNN, 4 seeds x 5 iters) ===", flush=True)
cfgv = FlywheelConfig(
    obs_mode="image", img_size=64, seed_demos=60, collect_per_iter=50,
    eval_starts=200, iterations=5, seeds=4,
    strategies=("none", "relabel", "relabel_curated", "relabel_balanced"),
    epochs=80, finetune_epochs=30, hidden=128,
    out_dir=str(OUT / "vision"), report_strategy="relabel_curated", verbose=True,
)
sv = run_flywheel(cfgv)
write_results(cfgv, sv, OUT / "vision")
print(f"VISION DONE in {time.time() - t0:.0f}s", flush=True)
print("ALL DONE", flush=True)
