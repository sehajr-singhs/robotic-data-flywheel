"""Smoke tests for the pixel-observation path (renderer, CNN policy, loop).

These keep the image size and epochs tiny so the suite stays fast on CPU;
the real vision study runs on GPU via the Kaggle pipeline.
"""

import numpy as np
import pytest

from datafly.envs.planar_pusher import PlanarPusher
from datafly.policies.expert import ScriptedExpert
from datafly.policies.mlp import collect_trajectories

torch = pytest.importorskip("torch")


def test_renderer_shapes_and_determinism():
    env = PlanarPusher(seed=0, record_images=True, img_size=64)
    env.reset()
    img1 = env.render()
    img2 = env.render()
    assert img1.shape == (64, 64, 3)
    assert img1.dtype == np.uint8
    assert 0 <= img1.min() and img1.max() <= 255
    assert np.array_equal(img1, img2)          # deterministic renderer
    assert not np.array_equal(img1, img1[0, 0])  # scene is non-trivial


def test_trajectory_records_images():
    expert = ScriptedExpert(noise=0.1, rng=np.random.default_rng(1))
    env = PlanarPusher(seed=1, record_images=True, img_size=32)
    demos = collect_trajectories(env, None, n=4, rng=np.random.default_rng(2),
                                 act_fn=expert.act_from_state, source="expert")
    for t in demos:
        assert t.images is not None
        assert t.images.shape[0] == len(t)          # one image per (state, action) pair
        assert t.images.shape[1:] == (32, 32, 3)


def test_cnn_trains_and_acts():
    from datafly.policies.cnn import CNNPolicy, train_bc_image

    expert = ScriptedExpert(noise=0.1, rng=np.random.default_rng(3))
    env = PlanarPusher(seed=2, record_images=True, img_size=32)
    demos = collect_trajectories(env, None, n=4, rng=np.random.default_rng(4),
                                 act_fn=expert.act_from_state, source="expert")
    pol = train_bc_image(demos, hidden=16, epochs=2, batch_size=8, img_size=32)
    a = pol.act(demos[0].images[0])
    assert a.shape == (2,)
    assert np.all(np.abs(a) <= 1.0 + 1e-6)
    # fine-tuning path (flywheel update)
    pol2 = train_bc_image(demos, hidden=16, epochs=1, batch_size=8, img_size=32, init=pol)
    assert pol2.act(demos[0].images[0]).shape == (2,)


def test_vision_loop_runs():
    from datafly import FlywheelConfig, run_flywheel

    cfg = FlywheelConfig(obs_mode="image", img_size=32, seed_demos=4,
                         collect_per_iter=3, eval_starts=6, iterations=1,
                         seeds=1, epochs=2, finetune_epochs=1, hidden=16,
                         strategies=("none", "relabel"), report_strategy="relabel",
                         out_dir="results/_test_vision", verbose=False)
    s = run_flywheel(cfg)
    assert s["config"]["obs_mode"] == "image"
    assert set(s["strategies"]) == {"none", "relabel"}
    for name in s["strategies"]:
        assert len(s["strategies"][name]["success_rate_mean"]) == 2  # iter 0 + 1
