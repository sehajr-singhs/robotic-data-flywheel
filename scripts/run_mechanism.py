"""Mechanism study: map the flywheel's stability region in
(capacity x relabel-mix-ratio) space.

The flood hypothesis: blind relabeling grows the dataset's relabeled:clean
ratio without bound and the policy overfits its own failure distribution.
This runner sweeps the two axes that should control the boundary --

    capacity  : hidden width of the BC policy (32 / 96 / 256 for the MLP,
                32 / 128 for the CNN)
    mix ratio : the controlled relabeled:clean ratio in the training set,
                held fixed by `relabel_mix` (0.1 ... 2.0)

-- plus the reference strategies (relabel, relabel_curated) and the
closed-loop controller (relabel_adaptive) at each capacity.

Layout of the output:

    results_mech/<obs>/cap<C>/<strategy-or-r<ratio>>/strategy_*.json + run_config.json
    results_mech/<obs>/manifest.json   (machine-readable grid for the analysis)

Usage:
    python scripts/run_mechanism.py --obs state --capacities 32 96 256 \
        --ratios 0.1 0.25 0.5 1 2 --seeds 3 --out results_mech/state
    python scripts/run_mechanism.py --obs image --capacities 32 128 \
        --ratios 0.25 0.5 1 --seeds 2 --quick
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datafly import FlywheelConfig, run_flywheel

REFERENCE_STRATEGIES = ("relabel", "relabel_curated", "relabel_adaptive")


def run_cell(
    cfg: FlywheelConfig,
    name: str,
    out_dir: Path,
    manifest: dict,
    label: str,
) -> None:
    print(f"\n=== {label}: {name} ===", flush=True)
    s = run_flywheel(cfg)
    cell = out_dir / name
    cell.mkdir(parents=True, exist_ok=True)
    for sname, data in s["strategies"].items():
        (cell / f"strategy_{sname}.json").write_text(json.dumps(data, indent=2))
    (cell / "run_config.json").write_text(json.dumps({
        "config": s["config"], "cell": label,
    }, indent=2))
    manifest[label] = {
        "strategies": {
            sname: {"success_rate_mean": data["success_rate_mean"],
                    "curation_log": data.get("curation_log")}
            for sname, data in s["strategies"].items()
        },
        "config": s["config"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", choices=["state", "image"], default="state")
    ap.add_argument("--capacities", nargs="+", type=int, default=[32, 96, 256])
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--out", default="results_mech")
    ap.add_argument("--quick", action="store_true", help="tiny CPU smoke run")
    args = ap.parse_args()

    out = Path(args.out) / args.obs
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"obs": args.obs, "capacities": args.capacities,
                      "ratios": args.ratios, "seeds": args.seeds,
                      "iterations": args.iterations, "cells": {}}

    for cap in args.capacities:
        if args.obs == "state":
            base = FlywheelConfig(
                seed_demos=100, collect_per_iter=80, eval_starts=300,
                iterations=args.iterations, seeds=args.seeds,
                hidden=cap, epochs=150, finetune_epochs=50,
                track_train_loss=True, verbose=False,
            )
        else:
            base = FlywheelConfig(
                obs_mode="image", img_size=64, seed_demos=40,
                collect_per_iter=30, eval_starts=120,
                iterations=min(args.iterations, 4), seeds=args.seeds,
                hidden=cap, epochs=60, finetune_epochs=20,
                track_train_loss=True, verbose=False,
            )
        if args.quick:
            base = base.quick()
            base.hidden = cap
            base.track_train_loss = True

        for name in REFERENCE_STRATEGIES:
            cfg = FlywheelConfig(**{**vars(base), "strategies": (name,),
                                    "out_dir": str(out)})
            run_cell(cfg, name, out, manifest["cells"], f"cap{cap}/{name}")

        for r in args.ratios:
            cfg = FlywheelConfig(**{**vars(base),
                                    "strategies": ("relabel_mix",),
                                    "strategy_kwargs": {"relabel_mix": {"mix_ratio": r}},
                                    "out_dir": str(out)})
            run_cell(cfg, f"r{r:g}", out, manifest["cells"], f"cap{cap}/mix{r:g}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {out / 'manifest.json'} ({len(manifest['cells'])} cells)")


if __name__ == "__main__":
    main()
