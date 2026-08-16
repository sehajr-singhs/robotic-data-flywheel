"""Honest cost accounting: flywheel vs tuned RL.

The naive comparison (flywheel frames vs DQN steps) ignores that the
flywheel spends *oracle labels* — human teleoperator queries that cost real
money — while DQN spends free simulation steps. This script makes the
accounting explicit with a label-price lambda:

    total_cost(strategy) = env_steps + lambda * oracle_queries

and reports the crossover lambda* where a *tuned* DQN beats the flywheel at
equal cost. If lambda* is large (labels are expensive), the flywheel's
advantage shrinks; the paper should report this honestly instead of the
one-sided "10x fewer interactions" claim.

Usage:
    python scripts/flywheel_vs_rl.py --main results_v3/main --dqn results_dqn/best.json
    python scripts/flywheel_vs_rl.py --quick
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LAMBDAS = (0.1, 1.0, 10.0, 100.0)


def flywheel_costs(summary: dict) -> dict:
    """Env steps (upper bound) + oracle queries for the relabel strategies."""
    cfg = summary["config"]
    horizon = cfg["horizon"]
    iters = cfg["iterations"]
    episodes = cfg["seed_demos"] + iters * cfg["collect_per_iter"]
    env_steps = episodes * horizon  # upper bound (episodes often end early)
    out = {}
    for name, data in summary["strategies"].items():
        log = data.get("curation_log") or []
        queries = int(sum(c.get("oracle_queries", 0) for c in log))
        out[name] = {"env_steps": env_steps, "oracle_queries": queries,
                     "final_success": data["success_rate_mean"][-1]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="results_v3/main/summary.json")
    ap.add_argument("--dqn", default="results_dqn/best_curve.json",
                    help="tuned DQN curve json (with 'eval' success-vs-steps)")
    ap.add_argument("--out", default="results/figs/cost_crossover.png")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    main = json.loads(Path(args.main).read_text())
    dqn = json.loads(Path(args.dqn).read_text())
    fc = flywheel_costs(main)

    if args.quick:
        dqn = {"eval": [{"env_steps": 100_000, "success_rate": 0.1},
                        {"env_steps": 300_000, "success_rate": 0.2}]}

    out_json = {"flywheel": fc, "lambdas": list(LAMBDAS), "dqn_final": None}
    # DQN success as a function of cost; use the best success at or below cost
    dqn_steps = [e["env_steps"] for e in dqn["eval"]]
    dqn_succ = [e["success_rate"] for e in dqn["eval"]]
    dqn_final = dqn_succ[-1]
    out_json["dqn_final"] = dqn_final

    for name in ("relabel", "relabel_curated"):
        fw = fc[name]
        fw_env, fw_q = fw["env_steps"], fw["oracle_queries"]
        fw_cost = {lam: fw_env + lam * fw_q for lam in LAMBDAS}
        # cost at which tuned DQN reaches the flywheel's final success
        crossover = {}
        for lam in LAMBDAS:
            budget = fw_cost[lam]
            dqn_at_budget = dqn_succ[-1] if dqn_steps[-1] <= budget else float(
                np.interp(budget, dqn_steps, dqn_succ))
            crossover[lam] = dqn_at_budget
        out_json[name] = {
            "env_steps": fw_env, "oracle_queries": fw_q,
            "final_success": fw["final_success"],
            "flywheel_cost": fw_cost,
            "dqn_success_at_flywheel_cost": crossover,
            "dqn_wins": {lam: crossover[lam] >= fw["final_success"] for lam in LAMBDAS},
        }

    # crossover lambda: cost-equal price of a label at which DQN first ties
    def dqn_success(cost: float) -> float:
        return float(np.interp(cost, dqn_steps, dqn_succ))

    lo, hi = 0.0, 1e4
    fw = fc["relabel"]
    for _ in range(60):  # bisect on lambda
        mid = (lo + hi) / 2
        if dqn_success(fw["env_steps"] + mid * fw["oracle_queries"]) >= fw["final_success"]:
            hi = mid
        else:
            lo = mid
    out_json["crossover_lambda_relabel"] = round((lo + hi) / 2, 1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.with_suffix(".json").write_text(json.dumps(out_json, indent=2))
    print(json.dumps(out_json, indent=2))
    print(f"wrote {out_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
