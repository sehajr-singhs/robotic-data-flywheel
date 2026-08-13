"""PlanarPusher: a minimal 2-link arm pushing a block to a target region.

A pure-numpy *kinematic* pusher. There is no inertia or contact solver: the
fingertip "captures" the block whenever it is inside a small disc around the
block, and the block then follows the tip (with slip) until contact is lost.
This is deliberately simple — it lets us run thousands of episodes per minute
on a laptop CPU, which is what makes full flywheel studies (collect -> curate
-> retrain -> repeat) tractable without a GPU farm. The task is still hard
enough to be interesting: behavior-cloned policies fail on a large fraction
of start configurations, and the *curation strategy* decides whether the
flywheel spins up or stalls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

L1 = 0.50  # link 1 length (base -> elbow)
L2 = 0.40  # link 2 length (elbow -> fingertip)

CAP_RADIUS = 0.055    # tip captures the block within this distance
STICK = 0.85          # how strongly the block follows the tip while captured
FRICTION = 0.30       # per-step block-velocity decay once contact is lost
SUCCESS_RADIUS = 0.07 # block must end within this of the target
SUCCESS_HOLD = 5      # steps the block must remain in the goal region
MAX_Q_DOT = 1.6       # rad / s
DT = 0.10             # s per step
DEFAULT_HORIZON = 70

STATE_DIM = 12
ACTION_DIM = 2


@dataclass
class Obs:
    """Full observation of the pusher state at a single timestep."""
    q: np.ndarray       # (2,) joint angles
    dq: np.ndarray      # (2,) joint velocities (last commanded action)
    block: np.ndarray   # (2,) block position
    dblock: np.ndarray  # (2,) block velocity
    target: np.ndarray  # (2,) target position
    tip: np.ndarray     # (2,) fingertip position (derived)


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """Fingertip position from joint angles."""
    q1, q2 = q
    return np.array([
        L1 * np.cos(q1) + L2 * np.cos(q1 + q2),
        L1 * np.sin(q1) + L2 * np.sin(q1 + q2),
    ])


def jacobian(q: np.ndarray) -> np.ndarray:
    """2x2 analytic Jacobian of the fingertip wrt joint angles."""
    q1, q2 = q
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    return np.array([
        [-L1 * np.sin(q1) - L2 * s12, -L2 * s12],
        [L1 * np.cos(q1) + L2 * c12,  L2 * c12],
    ])


def ik2(tip: np.ndarray, elbow: str = "down") -> np.ndarray:
    """Closed-form inverse kinematics for the 2R arm (elbow up or down)."""
    x, y = tip
    d = np.hypot(x, y)
    d = np.clip(d, abs(L1 - L2) + 1e-3, L1 + L2 - 1e-3)
    c2 = (d * d - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    c2 = np.clip(c2, -1.0, 1.0)
    s2 = np.sqrt(max(0.0, 1.0 - c2 * c2))
    if elbow == "down":
        s2 = -s2
    q2 = np.arctan2(s2, c2)
    q1 = np.arctan2(y, x) - np.arctan2(L2 * s2, L1 + L2 * c2)
    return np.array([q1, q2])


def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point p to segment ab."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def state_vector(o: Obs) -> np.ndarray:
    """Flat feature vector fed to policies and scoring functions."""
    return np.concatenate([
        o.q, o.dq, o.block, o.dblock, o.target, o.tip,
    ]).astype(np.float32)


class PlanarPusher:
    """The environment. Step with joint-velocity commands in [-1, 1]."""

    def __init__(
        self,
        seed: int = 0,
        horizon: int = DEFAULT_HORIZON,
        dt: float = DT,
        rng: Optional[np.random.Generator] = None,
    ):
        self.horizon = horizon
        self.dt = dt
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.t = 0
        self.obs: Optional[Obs] = None
        self.in_goal_steps = 0
        self._traj_states: list[np.ndarray] = []
        self._traj_actions: list[np.ndarray] = []

    # ------------------------------------------------------------------ #
    # sampling helpers (also used by the experiment harness)              #
    # ------------------------------------------------------------------ #
    def sample_start(self) -> Obs:
        """Random feasible start: block and target within reach of the arm."""
        # Block comfortably inside the arm's workspace (max reach is 0.9 m;
        # tasks near the boundary are *not* pushable — pushing radially
        # outward at full extension hits a Jacobian singularity).
        block = self.rng.uniform([0.38, -0.22], [0.60, 0.22])
        # Target within a ring around the block — but only *pushable* tasks:
        # the target must lie on the far side of the block from the arm base
        # (a block cannot be pulled toward the base by pushing). Rejection-
        # sample until the push direction points outward.
        best = None
        best_out = -np.inf
        for _ in range(500):
            ang = self.rng.uniform(0.0, 2 * np.pi)
            rad = self.rng.uniform(0.08, 0.18)
            target = block + rad * np.array([np.cos(ang), np.sin(ang)])
            target[0] = np.clip(target[0], 0.15, 0.75)
            target[1] = np.clip(target[1], -0.30, 0.30)
            push_dir = target - block
            pn = np.linalg.norm(push_dir)
            if pn < 1e-6:
                continue
            outward = np.dot(push_dir / pn, block / max(np.linalg.norm(block), 1e-6))
            if outward > 0.25:  # target is on the far side of the block
                best, best_out = target, outward
                break
            if outward > best_out:
                best, best_out = target, outward
        target = best if best is not None else target
        push_dir = target - block
        push_dir = push_dir / np.linalg.norm(push_dir)
        # Start the fingertip *just behind the block* (opposite the target,
        # at the capture-disc edge) so the task is: push. The policy never
        # has to learn a long approach phase — it must orient and push.
        tip = block - push_dir * 0.045 + self.rng.normal(0.0, 0.012, size=2)
        q = ik2(tip, elbow="down") + self.rng.normal(0.0, 0.03, size=2)
        q = np.clip(q, [-2.8, -2.4], [2.8, 2.4])
        return Obs(
            q=q,
            dq=np.zeros(2),
            block=block,
            dblock=np.zeros(2),
            target=target,
            tip=forward_kinematics(q),
        )

    def reset(self, obs: Optional[Obs] = None) -> Obs:
        self.t = 0
        self.in_goal_steps = 0
        self.obs = obs if obs is not None else self.sample_start()
        self._traj_states = [state_vector(self.obs)]
        self._traj_actions = []
        return self.obs

    def step(self, action: np.ndarray) -> Obs:
        assert self.obs is not None, "call reset() first"
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        q_dot = a * MAX_Q_DOT
        o = self.obs

        # --- integrate the arm ---------------------------------------- #
        tip_old = o.tip
        q_new = o.q + q_dot * self.dt
        tip_new = forward_kinematics(q_new)
        tip_vel = (tip_new - tip_old) / self.dt

        # --- block contact model -------------------------------------- #
        # Contact is detected along the whole tip path (the segment from the
        # previous to the new fingertip), not just at the endpoint — a fast
        # push sweeps *through* the block disc, and the block must ride with
        # it. Without this, one step of pushing exits the disc and the block
        # never moves at all.
        d_old = np.linalg.norm(tip_old - o.block)
        d_new = np.linalg.norm(tip_new - o.block)
        d_seg = _segment_distance(o.block, tip_old, tip_new)
        contact = min(d_old, d_new, d_seg) < CAP_RADIUS

        if contact:
            dblock_new = tip_vel * STICK
        else:
            dblock_new = o.dblock * FRICTION
        block_new = o.block + dblock_new * self.dt

        # --- goal check ------------------------------------------------ #
        in_goal = np.linalg.norm(block_new - o.target) < SUCCESS_RADIUS
        self.in_goal_steps = self.in_goal_steps + 1 if in_goal else 0

        self.obs = Obs(
            q=q_new,
            dq=q_dot,
            block=block_new,
            dblock=dblock_new,
            target=o.target,
            tip=tip_new,
        )
        self.t += 1
        self._traj_states.append(state_vector(self.obs))
        self._traj_actions.append(a.copy())
        return self.obs

    @property
    def success(self) -> bool:
        return self.in_goal_steps >= SUCCESS_HOLD

    @property
    def done(self) -> bool:
        return self.success or self.t >= self.horizon

    @property
    def final_dist(self) -> float:
        assert self.obs is not None
        return float(np.linalg.norm(self.obs.block - self.obs.target))

    def trajectory(self) -> dict:
        """(states, actions, info) of the current episode.

        The state list has one more frame than the action list (the initial
        state precedes the first action). Pair each action with the state
        that preceded it — the standard BC dataset layout.
        """
        states = np.stack(self._traj_states)
        actions = np.stack(self._traj_actions) if self._traj_actions else np.zeros((0, 2))
        if len(actions) and len(states) == len(actions) + 1:
            states = states[:-1]
        return {
            "states": states,
            "actions": actions,
            "success": bool(self.success),
            "final_dist": self.final_dist,
            "steps": self.t,
        }
