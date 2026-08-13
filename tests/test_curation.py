import numpy as np

from datafly.curation.scores import coverage, progress, smoothness
from datafly.curation.strategies import (
    near_miss,
    none,
    relabel,
    relabel_plus_success,
    success_coverage,
    success_only,
)
from datafly.policies.expert import ScriptedExpert
from datafly.policies.mlp import Trajectory


def _traj(success, final_dist, seed=0, states=None, actions=None):
    if states is None:
        states = np.zeros((10, 12), dtype=np.float32)
        states[:, 4] = np.linspace(0.5, 0.55, 10)  # block drifts right
        states[:, 8] = 0.6                          # target
    if actions is None:
        actions = np.ones((10, 2), dtype=np.float32) * 0.3
    return Trajectory(states, actions, success, final_dist, 10, seed=seed)


def test_none_and_success_only():
    trjs = [_traj(True, 0.04), _traj(False, 0.3), _traj(True, 0.02)]
    assert none(trjs) == []
    kept = success_only(trjs)
    assert len(kept) == 2 and all(t.success for t in kept)


def test_near_miss_filters_by_distance():
    trjs = [_traj(False, 0.10), _traj(False, 0.30), _traj(True, 0.04)]
    kept = near_miss(trjs)
    assert len(kept) == 1 and kept[0].final_dist == 0.10


def test_relabel_replaces_actions():
    expert = ScriptedExpert(noise=0.0)
    trj = _traj(False, 0.3)
    out = relabel([trj], expert=expert)
    assert len(out) == 1
    assert out[0].source == "relabeled"
    assert not np.allclose(out[0].actions, trj.actions)


def test_relabel_plus_success_keeps_success_labels():
    expert = ScriptedExpert(noise=0.0)
    ok = _traj(True, 0.04, seed=0)
    bad = _traj(False, 0.3, seed=1)
    out = relabel_plus_success([ok, bad], expert=expert)
    by_seed = {t.seed: t for t in out}
    assert np.allclose(by_seed[0].actions, ok.actions)   # kept as-is
    assert by_seed[1].source == "relabeled"


def test_success_coverage_prefers_novel():
    states_a = np.zeros((10, 12), dtype=np.float32)
    states_a[:, 4] = 0.4
    states_b = np.zeros((10, 12), dtype=np.float32)
    states_b[:, 4] = 0.9
    trjs = [_traj(True, 0.04, seed=0, states=states_a), _traj(True, 0.05, seed=1, states=states_b)]
    db = np.zeros((100, 12), dtype=np.float32)
    db[:, 4] = 0.41  # dataset hugs states_a -> b is the novel one
    kept = success_coverage(trjs, dataset_states=db, coverage_cap=1)
    assert len(kept) == 1 and kept[0].seed == 1


def test_scores_are_sane():
    trj = _traj(False, 0.3)
    assert progress(trj) <= trj.final_dist
    assert smoothness(trj) == 0.0  # constant actions
    assert coverage(trj, None) == float("inf")
