"""Assemble results/summary.json from per-strategy files and render figures.

    python scripts/merge_results.py [--out-dir results]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datafly.viz import plot_flywheel_report, plot_success_curves, plot_trajectories


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    out = Path(args.out_dir)

    meta = json.loads((out / "run_config.json").read_text())
    strategies = {}
    for f in sorted(out.glob("strategy_*.json")):
        strategies[f.name[len("strategy_") : -len(".json")]] = json.loads(f.read_text())
    report = {}
    report_path = out / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())

    summary = {
        "config": meta["config"],
        "seed_expert_success_rate": meta["seed_expert_success_rate"],
        "strategies": strategies,
        "report": report,
    }
    summary["config"]["strategies"] = list(strategies)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    figdir = out / "figs"
    figdir.mkdir(parents=True, exist_ok=True)
    plot_success_curves(summary, figdir / "success_vs_iteration.png")
    if report:
        plot_flywheel_report(report, figdir / "flywheel_report.png")
        plot_trajectories(report["plot_trajs"], figdir / "trajectories.png")

    # human-readable table (mean +/- std)
    lines = ["| strategy | " + " | ".join(f"iter {i}" for i in range(len(next(iter(strategies.values()))["success_rate_mean"]))) + " |",
             "|---|---|" * len(next(iter(strategies.values()))["success_rate_mean"])]
    for name, data in strategies.items():
        cells = " | ".join(f"{m:.2f}±{s:.2f}" for m, s in
                           zip(data["success_rate_mean"], data["success_rate_std"]))
        lines.append(f"| {name} | {cells} |")
    (out / "summary_table.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nMerged -> {out / 'summary.json'}; figures in {figdir}/")


if __name__ == "__main__":
    main()
