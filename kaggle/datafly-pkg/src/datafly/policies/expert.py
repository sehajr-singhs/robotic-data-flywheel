"""A scripted push expert that works from *any* state.

This matters. A data flywheel relabels deployment failures with expert
actions (DAgger-style), and on a real robot that expert is a human
teleoperator. Here the expert is a hand-written controller so the loop is
fully reproducible on a laptop; it plays the role of "the oracle labeler"
in the flywheel experiments.

The controller is deliberately simple and physical:
  * approach a stand-off point *behind* the block (opposite the target),
    with speed proportional to the distance remaining;
  * then push along the block->target line, slowing proportionally as the
    block nears the target (this is how a careful human pushes);
  * stop pushing inside a deadband and let friction settle the block in the
    goal region.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..envs.planar_pusher import (
    CAP_RADIUS,
    DT,
    MAX_Q_DOT,
    SUCCESS_RADIUS,
    forward_kinematics,
    jacobian,
    state_vector,
)

_BEHIND_DIST = 0.085     # how far behind the block the tip lines up
_APPROACH_SPEED_K = 4.0  # approach speed = k * distance to stand-off point
_PUSH_SPEED_K = 4.0      # push speed = k * distance from block to target
_MAX_TIP_SPEED = 1.2     # m / s
_STOP_DIST = 0.015       # deadband: stop pushing (block coasts a little)
_HOLD_RADIUS = 0.075      # hysteresis: keep holding until block drifts past
_REENTER_DIST = 0.045    # if contact lost, blend approach back in
_DAMP = 0.02             # DLS damping for the Jacobian pseudo-inverse


def _dls_inverse(J: np.ndarray, lam: float = _DAMP) -> np.ndarray:
    """Damped least-squares inverse: J^T (J J^T + lam I)^{-1}."""
    JJt = J @ J.T
    return J.T @ np.linalg.inv(JJt + lam * np.eye(2))


class ScriptedExpert:
    """Pushes the block to the target from any reachable state."""

    def __init__(self, noise: float = 0.0, rng: Optional[np.random.Generator] = None,
                 cap_radius: float = CAP_RADIUS):
        self.noise = noise
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.cap_radius = cap_radius  # contact range of the *simulator* (MuJoCo's
                                      # sphere+cylinder contact is 0.065, not 0.055)
        self._holding = False  # hysteresis: don't re-push once stopped in goal

    def act_from_state(self, s: np.ndarray) -> np.ndarray:
        q = s[0:2]
        block = s[4:6]
        target = s[8:10]
        tip = s[10:12]

        push_dir = target - block
        norm = np.linalg.norm(push_dir)
        if norm < 1e-6:
            push_dir = np.array([1.0, 0.0])
        else:
            push_dir = push_dir / norm

        # --- stop deadband with hysteresis: once the block is settled in
        # the goal, hold still until it drifts well outside (avoids the
        # re-push oscillation that keeps it from accumulating hold steps) -- #
        if norm <= _STOP_DIST:
            self._holding = True
            return np.zeros(2)
        if self._holding and norm < _HOLD_RADIUS:
            return np.zeros(2)
        if norm > _HOLD_RADIUS + 0.02:
            self._holding = False

        behind = block - push_dir * _BEHIND_DIST
        err = behind - tip
        dist_err = np.linalg.norm(err)

        if dist_err > 0.025:
            # --- approach the stand-off point, speed ~ distance --------- #
            speed = min(_MAX_TIP_SPEED, _APPROACH_SPEED_K * dist_err)
            des_tip_vel = (err / dist_err) * speed
        else:
            # --- push: speed ~ remaining distance to target ------------- #
            contact = np.linalg.norm(tip - block) < self.cap_radius
            speed = min(_MAX_TIP_SPEED, _PUSH_SPEED_K * norm)
            des_tip_vel = push_dir * speed
            if not contact:
                # lost contact while pushing: re-enter the disc, pushing
                reenter = block - tip
                rn = np.linalg.norm(reenter)
                if rn > 1e-6:
                    des_tip_vel = (reenter / rn) * 0.7 + push_dir * 0.4
                    des_tip_vel = des_tip_vel / np.linalg.norm(des_tip_vel) * speed

        # resolve into joint velocities with the damped inverse Jacobian
        J = jacobian(q)
        q_dot = _dls_inverse(J) @ des_tip_vel
        q_dot = np.clip(q_dot, -MAX_Q_DOT, MAX_Q_DOT)

        action = q_dot / MAX_Q_DOT
        if self.noise > 0.0:
            action = action + self.rng.normal(0.0, self.noise, size=2)
        return np.clip(action, -1.0, 1.0)

    def act(self, obs) -> np.ndarray:
        return self.act_from_state(state_vector(obs))


class PushCommitExpert(ScriptedExpert):
    """Contact-rich push controller: once in contact, commit to the push.

    The base expert alternates between approach and push phases on a fixed
    stand-off distance; in a *kinematic* capture model the block rides with
    the tip so short taps still move it. In contact-rich simulation the tip
    must sustain contact force, so this controller only retreats to the
    stand-off point when contact is genuinely lost — while touching the
    block it keeps driving along the push direction until the goal is
    reached. Same state-vector interface, so it plugs into the flywheel
    loop unchanged.
    """

    def act_from_state(self, s: np.ndarray) -> np.ndarray:
        q = s[0:2]
        block = s[4:6]
        target = s[8:10]
        tip = s[10:12]

        push_dir = target - block
        norm = np.linalg.norm(push_dir)
        if norm < 1e-6:
            push_dir = np.array([1.0, 0.0])
        else:
            push_dir = push_dir / norm

        # stop deadband with hysteresis (same as base expert)
        if norm <= _STOP_DIST:
            self._holding = True
            return np.zeros(2)
        if self._holding and norm < _HOLD_RADIUS:
            return np.zeros(2)
        if norm > _HOLD_RADIUS + 0.02:
            self._holding = False

        behind = block - push_dir * _BEHIND_DIST
        dist_err = np.linalg.norm(behind - tip)
        contact = np.linalg.norm(tip - block) < self.cap_radius

        # contact-rich pushing punishes violence: a fast off-center contact
        # flings the block. Cap the tip speed well below the kinematic env's.
        speed_cap = min(_MAX_TIP_SPEED, 0.55)
        if not contact and dist_err > 0.025:
            # genuinely lost contact: approach the stand-off point first
            speed = min(speed_cap, _APPROACH_SPEED_K * dist_err)
            des_tip_vel = ((behind - tip) / dist_err) * speed
        else:
            # in contact (or close): commit to the push
            speed = min(speed_cap, 2.5 * norm)
            des_tip_vel = push_dir * speed
            if not contact:
                # just outside contact: blend in re-entry, keep pushing
                reenter = block - tip
                rn = np.linalg.norm(reenter)
                if rn > 1e-6:
                    des_tip_vel = (reenter / rn) * 0.7 + push_dir * 0.4
                    des_tip_vel = des_tip_vel / np.linalg.norm(des_tip_vel) * speed * 0.5

        J = jacobian(q)
        q_dot = _dls_inverse(J) @ des_tip_vel
        q_dot = np.clip(q_dot, -MAX_Q_DOT, MAX_Q_DOT)
        action = q_dot / MAX_Q_DOT
        if self.noise > 0.0:
            action = action + self.rng.normal(0.0, self.noise, size=2)
        return np.clip(action, -1.0, 1.0)
