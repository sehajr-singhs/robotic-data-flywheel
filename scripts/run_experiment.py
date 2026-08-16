"""Run the flywheel experiment and write per-strategy results.

Each strategy writes results/strategy_<name>.json so the full comparison can
be spread across several invocations; scripts/merge_results.py assembles the
final results/summary.json + figures.

Usage:
    python scripts/run_experiment.py                     # all strategies
    python scripts/run_experiment.py --strategies relabel relabel_curated
    python scripts/run_experiment.py --quick             # smoke test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datafly import FlywheelConfig, run_flywheel


def main() -> None:
    ap = argparse.ArgumentParser(description="Data flywheel experiment")
    ap.add_argument("--quick", action="store_true", help="downscaled smoke run")
    ap.add_argument("--seed-demos", type=int)
    ap.add_argument("--collect-per-iter", type=int)
    ap.add_argument("--eval-starts", type=int)
    ap.add_argument("--iterations", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--finetune-epochs", type=int)
    ap.add_argument("--seeds", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--expert-noise", type=float)
    ap.add_argument("--oracle-noise", type=float)
    ap.add_argument("--success-radius", type=float)
    ap.add_argument("--target-ring", nargs=2, type=float, default=None,
                    metavar=("RMIN", "RMAX"))
    ap.add_argument("--horizon", type=int)
    ap.add_argument("--obs-mode", choices=["state", "image"], default=None)
    ap.add_argument("--img-size", type=int)
    ap.add_argument("--strategies", nargs="+", default=None)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--report-strategy", default="relabel_curated")
    args = ap.parse_args()

    cfg = FlywheelConfig(out_dir=args.out_dir, report_strategy=args.report_strategy)
    for k, v in vars(args).items():
        if v is not None and k not in ("quick", "out_dir", "report_strategy", "strategies",
                                       "target_ring"):
            setattr(cfg, k, v)
    if args.target_ring:
        cfg.target_ring = tuple(args.target_ring)
    if args.strategies:
        cfg.strategies = tuple(args.strategies)
    if args.quick:
        cfg = cfg.quick()

    summary = run_flywheel(cfg)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # per-strategy files so partial runs can be merged
    for name, data in summary["strategies"].items():
        (out / f"strategy_{name}.json").write_text(json.dumps(data, indent=2))
    if summary["report"]:
        (out / "report.json").write_text(json.dumps(summary["report"], indent=2))
    # config + seed info for the merge step
    (out / "run_config.json").write_text(json.dumps({
        "config": summary["config"],
        "seed_expert_success_rate": summary["seed_expert_success_rate"],
        "report_strategy": cfg.report_strategy,
    }, indent=2))

    print(f"\nWrote per-strategy results to {cfg.out_dir}/ (merge with scripts/merge_results.py)")


if __name__ == "__main__":
    main()
