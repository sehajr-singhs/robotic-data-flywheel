"""Evaluation harness: run a policy on a fixed set of held-out starts.

Every strategy and every flywheel iteration is evaluated on the *same* set
of starts, so success-rate differences are attributable to the flywheel, not
to evaluation noise.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .envs.planar_pusher import Obs, PlanarPusher, state_vector


def evaluate(
    env: PlanarPusher,
    policy,
    starts: list[Obs],
    horizon: Optional[int] = None,
    act_fn: Optional[Callable] = None,
    obs_fn: Optional[Callable] = None,
) -> dict:
    """Return summary metrics over the fixed held-out start set.

    `obs_fn` maps an observation to what the policy consumes (state vector by
    default; `env.render` for vision policies).
    """
    if act_fn is None:
        act_fn = policy.act
    if obs_fn is None:
        obs_fn = state_vector
    horizon = horizon or env.horizon

    results = []
    for start in starts:
        env.reset(start)
        while not env.done:
            env.step(act_fn(obs_fn(env.obs)))
        results.append({
            "success": bool(env.success),
            "final_dist": env.final_dist,
            "steps": env.t,
        })

    success_rate = float(np.mean([r["success"] for r in results]))
    mean_final_dist = float(np.mean([r["final_dist"] for r in results]))
    return {
        "success_rate": success_rate,
        "mean_final_dist": mean_final_dist,
        "n": len(results),
        "successes": int(round(success_rate * len(results))),
        "per_episode": results,
    }


def make_eval_starts(env: PlanarPusher, n: int, seed: int) -> list[Obs]:
    """Fixed, reproducible held-out start set, sampled with its own RNG."""
    rng = np.random.default_rng(seed)
    env.rng = rng  # sample_start draws from env.rng
    return [env.sample_start() for _ in range(n)]
