import numpy as np

from datafly.envs.planar_pusher import PlanarPusher, SUCCESS_RADIUS, forward_kinematics, ik2
from datafly.policies.expert import ScriptedExpert


def test_ik_roundtrip():
    for tip in [(0.6, 0.2), (0.5, -0.3), (0.7, 0.1)]:
        q = ik2(np.array(tip))
        recovered = forward_kinematics(q)
        assert np.allclose(recovered, tip, atol=1e-3)


def test_env_step_shape_and_termination():
    env = PlanarPusher(seed=0, horizon=50)
    env.reset()
    for _ in range(50):
        env.step(np.array([0.5, -0.5]))
    assert env.done
    traj = env.trajectory()
    assert traj["states"].shape[1] == 12
    assert traj["actions"].shape[1] == 2
    assert len(traj["states"]) == 50  # one state per action (initial excluded)


def test_expert_achieves_high_success():
    env = PlanarPusher(seed=1, horizon=120)
    expert = ScriptedExpert(noise=0.0)
    successes = 0
    for _ in range(60):
        env.reset()
        while not env.done:
            env.step(expert.act(env.obs))
        successes += env.success
    # clean expert should solve the overwhelming majority of feasible starts
    assert successes >= 54


def test_expert_works_from_any_state():
    """The oracle property the DAgger relabeling relies on."""
    env = PlanarPusher(seed=2, horizon=100)
    expert = ScriptedExpert(noise=0.0)
    env.reset()
    # jump the arm to an arbitrary configuration mid-episode
    env.obs.q = np.array([0.9, -1.2])
    env.obs.tip = forward_kinematics(env.obs.q)
    env.obs.dq = np.zeros(2)
    env.obs.dblock = np.zeros(2)
    for _ in range(120):
        env.step(expert.act(env.obs))
    # the oracle should recover from an arbitrary mid-episode state
    assert env.success or env.final_dist < 0.20


def test_block_physics_contact():
    """Tip inside the capture disc must move the block."""
    env = PlanarPusher(seed=3, horizon=10)
    start = env.sample_start()
    # teleport the tip onto the block
    start.block = np.array([0.5, 0.0])
    start.q = ik2(start.block + np.array([0.02, 0.0]))
    start.tip = forward_kinematics(start.q)
    env.reset(start)
    env.step(np.array([1.0, 0.0]))
    assert np.linalg.norm(env.obs.block - start.block) > 1e-4
