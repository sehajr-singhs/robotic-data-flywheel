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
relabel_balanced    : relabel everything but *cap the frames added per
                      iteration* at a fixed multiple of the current clean
                      dataset (the relabeled:clean ratio is the controlling
                      variable — high-capacity policies overfit a flood).
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


def relabel_balanced(
    trajs: list[Trajectory],
    expert: ScriptedExpert,
    dataset_states: Optional[np.ndarray] = None,
    **kwargs,
) -> list[Trajectory]:
    """Relabel everything, but cap added frames at a ratio of the current set.

    Blind DAgger floods the training set: each iteration adds ~collect_per_iter
    full rollouts (tens of thousands of frames), quickly overwhelming the
    clean demonstrations. `relabel_balanced` relabels every rollout but keeps
    at most `balance_ratio` x the *current* dataset size in new frames each
    iteration (keeping the newest, i.e. the most recent rollouts — closest to
    the current policy's distribution). This isolates the ratio as the
    controlling variable.
    """
    ratio = kwargs.get("balance_ratio", 1.0)
    if dataset_states is None:
        return relabel(trajs, expert)
    clean_frames = len(dataset_states)
    keep = max(1, int(ratio * clean_frames))
    out: list[Trajectory] = []
    acc = 0
    for t in trajs:  # newest rollouts first preserves policy-proximal data
        if acc + len(t) > keep:
            break
        out.append(_relabel(t, expert))
        acc += len(t)
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


# --------------------------------------------------------------------------- #
# Mix-ratio control: the mechanism study's strategies.                        #
#                                                                            #
# The crash hypothesis says blind relabeling floods the training set: the     #
# relabeled:clean frame ratio grows without bound and the policy overfits    #
# its own failure distribution. These strategies treat that ratio as the     #
# *controlled variable* and can also prune the existing dataset (returning   #
# a {"add", "drop"} dict instead of a plain list).                           #
# --------------------------------------------------------------------------- #

def relabel_mix(
    trajs: list[Trajectory],
    expert: ScriptedExpert,
    dataset: Optional[list[Trajectory]] = None,
    **kwargs,
) -> list[Trajectory] | dict:
    """DAgger labels with a hard cap on the dataset's relabeled:clean mix.

    Unlike `relabel_balanced` (which caps the *per-iteration addition* at a
    multiple of the growing dataset, so the flood still compounds), this
    strategy caps the *total* relabeled fraction of the training set at
    `mix_ratio` x the clean frames at all times. When the budget is exceeded
    the oldest relabeled trajectories are dropped (newest are policy-proximal,
    so they survive). Sweeping `mix_ratio` maps the flood boundary: the
    smallest ratio at which the loop still compounds vs. collapses.
    """
    ratio = kwargs.get("mix_ratio", 0.5)
    new = [_relabel(t, expert) for t in trajs]
    if dataset is None:
        return new
    clean = [t for t in dataset if t.source != "relabeled"]
    rel = [t for t in dataset if t.source == "relabeled"] + new
    budget = max(1, int(ratio * sum(len(t) for t in clean)))
    keep: list[Trajectory] = []
    acc = 0
    for t in reversed(rel):  # newest first — policy-proximal data survives
        if acc + len(t) > budget:
            break
        keep.append(t)
        acc += len(t)
    keep_set = set(map(id, keep))
    drop = [t for t in rel if id(t) not in keep_set]
    return {"add": new, "drop": drop}


def _make_relabel_adaptive(
    init_ratio: float = 1.0,
    grow: float = 1.5,
    shrink: float = 0.5,
    rmin: float = 0.1,
    rmax: float = 4.0,
):
    """Closed-loop curator: the flywheel report sets the next mix ratio.

    The controller holds the dataset's relabeled:clean ratio at `r` (via the
    same budget rule as `relabel_mix`) and adapts `r` from the policy's own
    measured trajectory: if held-out success regressed since the previous
    iteration (the overfitting signature), halve the ratio; if it improved,
    admit more. No capacity information is used — the loop finds its own
    stable operating point near the flood boundary.

    Returns a stateful callable, one per (strategy, seed): pass `history`
    (the per-seed eval series) so the rule can react to the last two points.
    """
    state = {"ratio": init_ratio, "last": None}

    def _call(
        trajs: list[Trajectory],
        expert: ScriptedExpert,
        dataset: Optional[list[Trajectory]] = None,
        history: Optional[list[float]] = None,
        **kwargs,
    ) -> list[Trajectory] | dict:
        if history and len(history) >= 2:
            if history[-1] < history[-2] - 1e-6:
                state["ratio"] = max(rmin, state["ratio"] * shrink)
            else:
                state["ratio"] = min(rmax, state["ratio"] * grow)
        kwargs["mix_ratio"] = state["ratio"]
        out = relabel_mix(trajs, expert, dataset=dataset, **kwargs)
        out["ratio"] = state["ratio"]  # type: ignore[union-attr]
        return out

    return _call


STRATEGIES: dict[str, Callable[..., list[Trajectory]]] = {
    "none": none,
    "success_only": success_only,
    "near_miss": near_miss,
    "relabel": relabel,
    "relabel_plus_success": relabel_plus_success,
    "relabel_curated": relabel_curated,
    "relabel_balanced": relabel_balanced,
    "relabel_mix": relabel_mix,
    "success_coverage": success_coverage,
}

# Stateful strategies: factory -> fresh instance per (strategy, seed).
STRATEGY_MAKERS: dict[str, Callable[[], Callable[..., list[Trajectory]]]] = {
    "relabel_adaptive": lambda: _make_relabel_adaptive(),
}
