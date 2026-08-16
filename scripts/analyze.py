"""Ablation analysis: oracle quality crossover, label efficiency, difficulty.

Reads results/{main,oracle,hard}/ and writes:
  results/figs/oracle_crossover.png      final success vs oracle noise
  results/figs/label_efficiency.png      success vs cumulative oracle queries
  results/figs/difficulty.png            hard-task success curves
  results/ablations.json                 machine-readable numbers for the paper

Usage: python scripts/analyze.py [--results-dir results]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"relabel": "#dc2626", "relabel_curated": "#16a34a", "none": "#888888",
          "success_only": "#2f6fb3"}


def _last(series) -> float:
    return series["success_rate_mean"][-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--main-dir", default=None, help="v3 main study dir (results_v3/main)")
    ap.add_argument("--vision-dir", default=None, help="v3 vision study dir (results_v3/vision)")
    ap.add_argument("--dqn-file", default=None, help="v3 DQN baseline json")
    args = ap.parse_args()
    res = Path(args.results_dir)
    (res / "figs").mkdir(parents=True, exist_ok=True)

    out: dict = {"oracle_crossover": {}, "label_efficiency": {}, "difficulty": {}}

    # ---- 1. oracle-quality crossover ------------------------------------ #
    noises = [0.0, 0.2, 0.35]
    noise_dirs = {0.0: "n000", 0.2: "n020", 0.35: "n035"}
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for name in ("relabel", "relabel_curated"):
        finals, stds = [], []
        for n in noises:
            f = res / "oracle" / noise_dirs[n] / f"strategy_{name}.json"
            d = json.loads(f.read_text())
            finals.append(_last(d))
            stds.append(d["success_rate_std"][-1])
        ax.errorbar(noises, finals, yerr=stds, marker="o", capsize=3, lw=2,
                    color=COLORS[name], label=name.replace("_", " "))
        out["oracle_crossover"][name] = [round(v, 3) for v in finals]
    ax.set_xlabel("oracle labeling noise (action std)")
    ax.set_ylabel("final held-out success")
    ax.set_ylim(0.0, 0.8)
    ax.set_xticks(noises)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(res / "figs" / "oracle_crossover.png", dpi=160)
    plt.close(fig)

    # ---- 2. label efficiency (main run) ---------------------------------- #
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    main_path = (Path(args.main_dir) / "summary.json") if args.main_dir else res / "main" / "summary.json"
    main = json.loads(main_path.read_text())
    for name in ("relabel", "relabel_curated", "success_only"):
        d = main["strategies"][name]
        queries, succ = [], []
        cum = 0
        for i, (m, q) in enumerate(zip(d["success_rate_mean"], [0] + [c["oracle_queries"] for c in d["curation_log"]])):
            cum += q
            queries.append(cum)
            succ.append(m)
        ax.plot(queries, succ, marker="o", lw=2, color=COLORS.get(name, "#333"),
                label=name.replace("_", " "))
        out["label_efficiency"][name] = {"queries": queries, "success": [round(s, 3) for s in succ]}
    ax.set_xlabel("cumulative oracle queries")
    ax.set_ylabel("held-out success")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(res / "figs" / "label_efficiency.png", dpi=160)
    plt.close(fig)

    # ---- 3. difficulty robustness ---------------------------------------- #
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for name in ("none", "success_only", "relabel"):
        f = res / "hard" / f"strategy_{name}.json"
        d = json.loads(f.read_text())
        iters = list(range(len(d["success_rate_mean"])))
        ax.plot(iters, d["success_rate_mean"], marker="o", lw=2,
                color=COLORS.get(name, "#333"), label=name.replace("_", " "))
        out["difficulty"][name] = d["success_rate_mean"]
    ax.set_xlabel("flywheel iteration (hard task)")
    ax.set_ylabel("held-out success")
    ax.set_ylim(0.0, 0.6)
    ax.set_xticks(iters)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(res / "figs" / "difficulty.png", dpi=160)
    plt.close(fig)

    (res / "ablations.json").write_text(json.dumps(out, indent=2))
    print("wrote results/ablations.json + results/figs/{oracle_crossover,label_efficiency,difficulty}.png")

    # ---- v3: perception study + label-efficiency budget plot ------------- #
    from datafly.viz import (plot_budget_comparison, plot_sample_images,
                             plot_success_curves)

    main_dir = Path(args.main_dir) if args.main_dir else res / "v3" / "main"
    vis_dir = Path(args.vision_dir) if args.vision_dir else res / "v3" / "vision"
    dqn_path = Path(args.dqn_file) if args.dqn_file else res / "v3" / "dqn" / "dqn.json"

    if (main_dir / "summary.json").exists():
        v3_main = json.loads((main_dir / "summary.json").read_text())
        dqn = json.loads(dqn_path.read_text()) if dqn_path.exists() else {}
        plot_budget_comparison(v3_main, dqn, res / "figs" / "budget_comparison.png")
        print("wrote results/figs/budget_comparison.png (flywheel vs DQN)")
    if (vis_dir / "summary.json").exists():
        v3_vis = json.loads((vis_dir / "summary.json").read_text())
        plot_success_curves(v3_vis, res / "figs" / "vision_curves.png")
        # a few raw pixel observations the CNN sees
        from datafly.envs.planar_pusher import PlanarPusher
        from datafly.eval import make_eval_starts
        env = PlanarPusher(seed=0, img_size=v3_vis["config"]["img_size"])
        starts = make_eval_starts(env, 4, seed=1)
        plot_sample_images(env, starts, res / "figs" / "sample_observations.png")
        print("wrote results/figs/vision_curves.png + sample_observations.png")

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
