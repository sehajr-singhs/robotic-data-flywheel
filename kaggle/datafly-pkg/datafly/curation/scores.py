"""Trajectory scoring: the measurable signals a flywheel can act on.

A real industrial flywheel cannot just ask "did it succeed?" — it needs
cheap, automatic signals that predict *learnability*: how close did a failed
episode come, how smooth was the control, how much new state coverage does
this episode add to the training set. These scores are what the curation
strategies and the flywheel report are built on.
"""

from __future__ import annotations

import numpy as np

from ..policies.mlp import Trajectory

_FRAME_STRIDE = 3       # downsample frames when computing coverage
_MAX_DB_FRAMES = 4000   # cap on dataset frames used for NN search


def final_dist(t: Trajectory) -> float:
    return t.final_dist


def progress(t: Trajectory) -> float:
    """Best (minimum) distance from block to target achieved during the episode."""
    if t.steps == 0:
        return float(t.final_dist)
    block = t.states[:, 4:6]
    target = t.states[:, 8:10]
    return float(np.min(np.linalg.norm(block - target, axis=1)))


def smoothness(t: Trajectory) -> float:
    """Mean absolute action-to-action change (lower = smoother commands)."""
    if len(t.actions) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(t.actions, axis=0))))


def coverage(t: Trajectory, dataset_states: np.ndarray) -> float:
    """Mean min distance from this trajectory's frames to the dataset.

    High coverage = this episode visits states the model has rarely seen,
    which is exactly the signal a flywheel uses to decide that a deployment
    episode is worth labeling and feeding back.
    """
    if dataset_states is None or len(dataset_states) == 0:
        return float("inf")
    frames = t.states[::_FRAME_STRIDE]
    db = dataset_states[:: max(1, len(dataset_states) // _MAX_DB_FRAMES)]
    # chunked pairwise NN to keep memory flat
    mins = np.empty(len(frames))
    for i in range(0, len(frames), 64):
        chunk = frames[i : i + 64]
        d = np.linalg.norm(chunk[:, None, :] - db[None, :, :], axis=2)
        mins[i : i + len(chunk)] = d.min(axis=1)
    return float(np.mean(mins))


def score_all(
    trajs: list[Trajectory],
    dataset_states: np.ndarray,
) -> dict[int, dict[str, float]]:
    """Score every trajectory with the full signal set."""
    out = {}
    for t in trajs:
        out[id(t)] = {
            "success": float(t.success),
            "final_dist": final_dist(t),
            "progress": progress(t),
            "smoothness": smoothness(t),
            "coverage": coverage(t, dataset_states),
        }
    return out
