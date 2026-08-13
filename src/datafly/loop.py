"""The flywheel loop.

    collect deployment rollouts
        -> score them (success / progress / smoothness / coverage)
        -> curate (the strategy decides what goes back in)
        -> retrain on seed + curated data
        -> evaluate on held-out starts
        -> repeat

This is the whole thesis: a loop is only a *flywheel* if the data that comes
back makes the next loop better. The driver below is deliberately
policy-agnostic — swap the MLP for a VLA model and the loop still runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .curation.scores import coverage, final_dist, progress, smoothness
from .curation.strategies import STRATEGIES
from .envs.planar_pusher import PlanarPusher
from .eval import evaluate, make_eval_starts
from .policies.expert import ScriptedExpert
from .policies.mlp import collect_trajectories, train_bc

DEFAULT_STRATEGIES = (
    "none",
    "success_only",
    "near_miss",
    "relabel",
    "relabel_curated",
    "success_coverage",
)


@dataclass
class FlywheelConfig:
    seed_demos: int = 80
    expert_noise: float = 0.12
    collect_per_iter: int = 60
    eval_starts: int = 200
    iterations: int = 5
    strategies: tuple[str, ...] = DEFAULT_STRATEGIES
    seeds: int = 4               # independent training seeds, results averaged
    epochs: int = 150            # BC epochs for the seed model
    finetune_epochs: int = 50    # per-iteration fine-tuning epochs (flywheel)
    lr: float = 1e-3
    hidden: int = 96
    horizon: int = 70
    seed: int = 0
    oracle_noise: float = 0.0    # noise on the *relabeling* oracle (human-label error)
    success_radius: float = 0.07  # task difficulty: goal tolerance
    target_ring: tuple[float, float] = (0.08, 0.18)  # task difficulty: push distance
    report_strategy: str = "relabel_curated"
    out_dir: str = "results"
    verbose: bool = True

    def quick(self) -> "FlywheelConfig":
        """Downscaled config for smoke tests / CI."""
        return FlywheelConfig(
            seed_demos=12,
            expert_noise=self.expert_noise,
            collect_per_iter=12,
            eval_starts=15,
            iterations=2,
            strategies=self.strategies,
            epochs=30,
            lr=self.lr,
            hidden=64,
            seeds=2,
            horizon=self.horizon,
            seed=self.seed,
            report_strategy=self.report_strategy,
            out_dir=self.out_dir,
            verbose=False,
        )


def _dataset_states(dataset: list) -> np.ndarray:
    return np.concatenate([t.states for t in dataset], axis=0)


def run_flywheel(cfg: FlywheelConfig) -> dict:
    env = PlanarPusher(seed=cfg.seed, horizon=cfg.horizon,
                       success_radius=cfg.success_radius, target_ring=cfg.target_ring)
    expert = ScriptedExpert(noise=cfg.expert_noise, rng=np.random.default_rng(cfg.seed + 7))
    # The *relabeling* oracle is a separate expert so its quality (labeling
    # noise, the human-teleoperator analogue) can be varied independently.
    oracle = ScriptedExpert(noise=cfg.oracle_noise, rng=np.random.default_rng(cfg.seed + 17))

    # --- seed data: noisy expert demonstrations ---------------------- #
    seed_rng = np.random.default_rng(cfg.seed + 1)
    seed_demos = collect_trajectories(
        env,
        None,
        n=cfg.seed_demos,
        rng=seed_rng,
        act_fn=expert.act_from_state,
        source="expert",
        seed_base=0,
    )

    # --- fixed held-out evaluation starts ---------------------------- #
    eval_starts = make_eval_starts(env, cfg.eval_starts, seed=cfg.seed + 2)

    summary: dict = {
        "config": {
            "seed_demos": cfg.seed_demos,
            "expert_noise": cfg.expert_noise,
            "collect_per_iter": cfg.collect_per_iter,
            "eval_starts": cfg.eval_starts,
            "iterations": cfg.iterations,
            "seeds": cfg.seeds,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "hidden": cfg.hidden,
            "horizon": cfg.horizon,
            "seed": cfg.seed,
            "oracle_noise": cfg.oracle_noise,
            "success_radius": cfg.success_radius,
            "target_ring": list(cfg.target_ring),
            "strategies": list(cfg.strategies),
        },
        "seed_expert_success_rate": float(np.mean([d.success for d in seed_demos])),
        "strategies": {},
        "report": {},
    }

    for si, name in enumerate(cfg.strategies):
        if cfg.verbose:
            print(f"\n=== strategy: {name} ===")
        all_series: list[list[float]] = []
        curation_log: list[dict] = []
        report: dict = {}

        for s in range(cfg.seeds):
            rng_s = np.random.default_rng(cfg.seed + 100 * s)
            s_demos = collect_trajectories(
                env, None, n=cfg.seed_demos, rng=rng_s,
                act_fn=expert.act_from_state, source="expert",
            )
            dataset = list(s_demos)
            policy = train_bc(dataset, hidden=cfg.hidden, epochs=cfg.epochs,
                              lr=cfg.lr, seed=cfg.seed + si + s)
            series = [evaluate(env, policy, eval_starts)["success_rate"]]

            for it in range(1, cfg.iterations + 1):
                roll_rng = np.random.default_rng(cfg.seed + 1000 * si + 100 * s + it)
                rollouts = collect_trajectories(
                    env, policy, n=cfg.collect_per_iter, rng=roll_rng,
                    act_fn=policy.act, source="policy", seed_base=it * 1000,
                )
                db = _dataset_states(dataset)
                curated = STRATEGIES[name](rollouts, expert=oracle, dataset_states=db)
                dataset.extend(curated)

                # Flywheel update: fine-tune from the previous policy rather
                # than retraining from scratch — how real deployment loops
                # update, keeping training time flat as data grows. The
                # `none` strategy freezes the policy: no feedback, no update.
                if name != "none":
                    policy = train_bc(
                        dataset, hidden=cfg.hidden, epochs=cfg.finetune_epochs,
                        lr=cfg.lr, seed=cfg.seed + si + s, init=policy,
                    )
                series.append(evaluate(env, policy, eval_starts)["success_rate"])

                if s == 0:
                    curation_log.append({
                        "iteration": it,
                        "n_rollouts": len(rollouts),
                        "n_success": int(sum(r.success for r in rollouts)),
                        "n_curated": len(curated),
                        "n_relabeled": int(sum(t.source == "relabeled" for t in curated)),
                        "oracle_queries": int(sum(len(t) for t in curated if t.source == "relabeled")),
                        "dataset_frames": sum(len(t) for t in dataset),
                        "rollout_success_rate": float(np.mean([r.success for r in rollouts])),
                    })
                    if cfg.verbose:
                        c = curation_log[-1]
                        print(
                            f"  seed {s} iter {it}: eval {series[-1]:.2f} "
                            f"| rollout {c['rollout_success_rate']:.2f} "
                            f"| curated {c['n_curated']} / {c['n_rollouts']}"
                        )

            all_series.append(series)

            # keep the last batch of rollouts from the report strategy (seed 0)
            if name == cfg.report_strategy and s == 0 and it == cfg.iterations:
                first_success = next((r for r in rollouts if r.success), rollouts[0])
                first_fail = next((r for r in rollouts if not r.success), rollouts[-1])
                report = {
                    "iteration": cfg.iterations,
                    "rollouts": [
                        {
                            "seed": r.seed,
                            "success": r.success,
                            "final_dist": final_dist(r),
                            "progress": progress(r),
                            "smoothness": smoothness(r),
                            "coverage": coverage(r, db),
                        }
                        for r in rollouts
                    ],
                    "plot_trajs": {
                        "expert seed": {
                            "states": s_demos[0].states.tolist(),
                            "success": s_demos[0].success,
                            "final_dist": s_demos[0].final_dist,
                        },
                        "policy iter 1 (fail)": {
                            "states": first_fail.states.tolist(),
                            "success": first_fail.success,
                            "final_dist": first_fail.final_dist,
                        },
                        "policy final (success)": {
                            "states": first_success.states.tolist(),
                            "success": first_success.success,
                            "final_dist": first_success.final_dist,
                        },
                    },
                }

        arr = np.array(all_series)
        summary["strategies"][name] = {
            "success_rate_mean": [round(float(v), 4) for v in arr.mean(axis=0)],
            "success_rate_std": [round(float(v), 4) for v in arr.std(axis=0)],
            "success_rate_seeds": [[round(float(v), 4) for v in row] for row in arr],
            "curation_log": curation_log,
        }
        if name == cfg.report_strategy:
            summary["report"] = report

    return summary


def save_results(summary: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    return path
