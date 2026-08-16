"""Tune the DQN anchor so the RL baseline is actually competitive.

The committed DQN (0.045 success after 300k steps) was barely trained: no
double-Q, a single hyperparameter point. A lopsided baseline makes the
flywheel-vs-RL comparison cheap. This sweep tunes the anchor properly and
writes the full success-vs-steps curve for the best config, so the paper
compares the flywheel against *tuned* RL, not a strawman.

Usage:
    python scripts/tune_dqn.py --budget 400000 --seeds 2 --out results_dqn
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

from datafly.envs.planar_pusher import PlanarPusher
from datafly.eval import make_eval_starts
from datafly.policies.dqn import train_dqn

GRID = {
    "lr": [3e-4, 1e-3],
    "hidden": [128, 256],
    "target_every": [1_000, 2_000],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=400_000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--eval-starts", type=int, default=200)
    ap.add_argument("--out", default="results_dqn")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = PlanarPusher(seed=0)
    starts = make_eval_starts(env, args.eval_starts, seed=2)

    results: dict = {"configs": {}, "curves": {}}
    best = None
    for lr, hidden, target_every in product(*GRID.values()):
        if args.quick:
            continue  # --quick just writes the skeleton
        key = f"lr{lr:g}_h{hidden}_te{target_every}"
        seeds_out = []
        for seed in range(args.seeds):
            print(f"\n=== DQN {key} seed {seed} ===", flush=True)
            r = train_dqn(env, starts, budget=args.budget, seed=seed,
                          hidden=hidden, lr=lr, target_every=target_every,
                          eval_every=25_000, verbose=True)
            seeds_out.append(r)
            (out / f"{key}_seed{seed}.json").write_text(json.dumps(r, indent=2))
        finals = [s["final_success"] for s in seeds_out]
        mean_final = float(np.mean(finals))
        results["configs"][key] = {
            "lr": lr, "hidden": hidden, "target_every": target_every,
            "final_success_mean": mean_final,
            "final_success_seeds": finals,
        }
        results["curves"][key] = seeds_out[0]["eval"]
        if best is None or mean_final > best["final_success_mean"]:
            best = {**results["configs"][key], "key": key}
        print(f"  {key}: final {mean_final:.3f}", flush=True)

    results["best"] = best
    results["budget"] = args.budget
    (out / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nbest config: {best}\nwrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
