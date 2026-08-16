"""Reassemble a run_flywheel summary.json from kernel per-strategy files.

The Kaggle kernels write one strategy_<name>.json per strategy (plus
report.json + run_config.json) instead of a single summary.json. This
rebuilds the summary the analysis/render pipeline expects:

    python scripts/rebuild_summary.py <dir> [<dir2> ...]

Each <dir> must contain strategy_*.json (+ optional report.json,
run_config.json). Writes <dir>/summary.json.
"""

import json
import sys
from pathlib import Path


def rebuild(directory: Path) -> Path:
    out = {}
    strategies = {}
    for f in sorted(directory.glob("strategy_*.json")):
        name = f.stem[len("strategy_"):]
        strategies[name] = json.loads(f.read_text())
    out["strategies"] = strategies
    rc = directory / "run_config.json"
    if rc.exists():
        data = json.loads(rc.read_text())
        out["config"] = data.get("config", {})
        out["seed_expert_success_rate"] = data.get("seed_expert_success_rate")
        out["report_strategy"] = data.get("report_strategy")
    rp = directory / "report.json"
    if rp.exists():
        out["report"] = json.loads(rp.read_text())
    else:
        out["report"] = {}
    dest = directory / "summary.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest} ({len(strategies)} strategies: {', '.join(strategies)})")
    return dest


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for d in sys.argv[1:]:
        rebuild(Path(d))


if __name__ == "__main__":
    main()
