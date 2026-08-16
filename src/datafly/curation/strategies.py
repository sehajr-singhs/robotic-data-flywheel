"""Curation strategies: the flywheel's decision rule for what goes back in.

Each strategy takes a batch of deployment rollouts and returns the subset
(possibly with modified labels) that gets added to the training set. The
experiment's central question is whether the choice of strategy determines
whether the flywheel spins up — and the answer turns out to be yes.

Strategies
----------
none                : no feedback at all (the flywheel never turns).
success_only        : add episodes that succeeded, labels untouched.
near_miss           : add failed episodes that came close, labels untouched
                      (tests whether *unlabeled* failure data helps).
relabel             : DAgger-style — relabel every collected state with the
                      expert's action before adding (the "oracle labeler"
                      approximation; on a robot this is a human operator).
relabel_plus_success: relabel failures, keep successful episodes as-is.
relabel_curated     : relabel failures *only if they made progress* (the
                      deployment signal says they were close); skip failures
                      that never got anywhere. The thesis strategy.
success_coverage    : among successes, keep the most *novel* ones by state
                      coverage, capped at k per iteration.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..policies.expert import ScriptedExpert
from ..policies.mlp import Trajectory
from .scores import coverage

NEAR_MISS_DIST = 0.15   # failed episodes within this final distance count as "close"
COVERAGE_CAP = 40       # max episodes kept by success_coverage per iteration
PROGRESS_THRESHOLD = 0.10  # relabel failures that ever got within this of the target


def none(trajs: list[Trajectory], **_) -> list[Trajectory]:
    return []


def success_only(trajs: list[Trajectory], **_) -> list[Trajectory]:
    return [t for t in trajs if t.success]


def near_miss(trajs: list[Trajectory], **kwargs) -> list[Trajectory]:
    threshold = kwargs.get("near_miss_dist", NEAR_MISS_DIST)
    return [t for t in trajs if not t.success and t.final_dist < threshold]


def _relabel(t: Trajectory, expert: ScriptedExpert) -> Trajectory:
    acts = np.array([expert.act_from_state(s) for s in t.states])
    return Trajectory(
        states=t.states,
        actions=acts,
        success=t.success,
        final_dist=t.final_dist,
        steps=t.steps,
        seed=t.seed,
        source="relabeled",
        images=t.images,  # pixels are preserved; only the *label* is replaced
    )


def relabel(trajs: list[Trajectory], expert: ScriptedExpert, **_) -> list[Trajectory]:
    """DAgger: relabel everything with the expert. No episode is thrown away."""
    return [_relabel(t, expert) for t in trajs]


def relabel_plus_success(
    trajs: list[Trajectory], expert: ScriptedExpert, **_
) -> list[Trajectory]:
    """Relabel failures, keep successful episodes' own (executed) actions."""
    out = []
    for t in trajs:
        if t.success:
            out.append(t)
        else:
            out.append(_relabel(t, expert))
    return out


def relabel_curated(
    trajs: list[Trajectory],
    expert: ScriptedExpert,
    dataset_states: Optional[np.ndarray] = None,
    **kwargs,
) -> list[Trajectory]:
    """Relabel only the failures that came close; keep successes as-is.

    Blindly ingesting every failed rollout floods the training set with
    far-off drifted states and can *hurt* a small model (state-distribution
    collapse). Curating by the progress signal — how close the episode ever
    got to the goal — keeps the corrective signal while dropping the garbage.
    """
    from .scores import progress

    threshold = kwargs.get("progress_threshold", PROGRESS_THRESHOLD)
    out = []
    for t in trajs:
        if t.success:
            out.append(t)
        elif progress(t) < threshold:
            out.append(_relabel(t, expert))
    return out


def success_coverage(
    trajs: list[Trajectory],
    dataset_states: np.ndarray,
    **kwargs,
) -> list[Trajectory]:
    """Keep the most novel successful episodes (capped per iteration)."""
    cap = kwargs.get("coverage_cap", COVERAGE_CAP)
    successes = [t for t in trajs if t.success]
    if not successes:
        return []
    scored = sorted(
        successes,
        key=lambda t: coverage(t, dataset_states),
        reverse=True,
    )
    return scored[:cap]


STRATEGIES: dict[str, Callable[..., list[Trajectory]]] = {
    "none": none,
    "success_only": success_only,
    "near_miss": near_miss,
    "relabel": relabel,
    "relabel_plus_success": relabel_plus_success,
    "relabel_curated": relabel_curated,
    "success_coverage": success_coverage,
}
