"""MuJoCoPusher: contact-rich version of PlanarPusher.

The numpy PlanarPusher is kinematic: the fingertip "captures" the block with
a synthetic stick-slip model. This env replaces that with real contact
dynamics in MuJoCo — the arm pushes the block through actual collision
(cylinder vs fingertip sphere on a frictional plane), so the block can slip,
rotate, or stall, and contact must be maintained across pushes. The
observation semantics are identical to PlanarPusher (q, dq, block, dblock,
target, tip -> the same 12-dim state vector), so the expert, the BC
policies, the curation strategies, and the flywheel loop all work unchanged.

Requires: pip install mujoco   (the modern mujoco package, any OS)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .planar_pusher import (
    CAP_RADIUS,
    DT,
    IMG_SIZE,
    L1,
    L2,
    MAX_Q_DOT,
    SUCCESS_HOLD,
    SUCCESS_RADIUS,
    Obs,
    forward_kinematics,
    ik2,
    state_vector,
)

MJCF = """
<mujoco model="arm_pusher">
  <compiler angle="radian"/>
  <option timestep="0.01" iterations="50" tolerance="1e-9" gravity="0 0 -9.81"/>
  <worldbody>
    <light name="sun" pos="0.4 0 2.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="table" type="plane" pos="0 0 0" size="1.2 1.2 0.01"
          friction="0.9 0.005 0.0001"/>
    <camera name="top" pos="0.4 0 2.2" xyaxes="1 0 0 0 1 0" fovy="36"/>
    <body name="base" pos="0 0 0.025">
      <joint name="j1" type="hinge" axis="0 0 1" limited="true" range="-2.8 2.8"/>
      <geom name="link1" type="capsule" fromto="0 0 0.025 0.5 0 0.025"
            size="0.020" mass="0.6" friction="0.5"/>
      <body name="elbow" pos="0.5 0 0.025">
        <joint name="j2" type="hinge" axis="0 0 1" limited="true" range="-2.4 2.4"/>
        <geom name="link2" type="capsule" fromto="0 0 0.025 0.4 0 0.025"
              size="0.018" mass="0.5" friction="0.5"/>
        <body name="tip" pos="0.4 0 0.030">
          <!-- pusher friction must be LOWER than block-table friction, or the
               block sticks to the pusher and gets dragged back on retreat
               (the push-slip oscillation that stalls contact-rich pushing). -->
          <geom name="fingertip" type="sphere" size="0.020" mass="0.02"
                friction="0.35 0.005 0.0001"/>
        </body>
      </body>
    </body>
    <body name="block" pos="0.5 0 0.03">
      <freejoint name="block_free"/>
      <!-- a flat-faced box: the fingertip sphere grips a plane face, so the
           push normal is stable and off-center contact does not roll the
           tip around a curved surface (the cylinder case). -->
      <geom name="block_geom" type="box" size="0.045 0.045 0.030" mass="0.6"
            friction="0.8 0.07 0.0001"/>
    </body>
    <body name="target_marker" pos="0.6 0 0.012">
      <joint name="mx" type="slide" axis="1 0 0" limited="false"/>
      <joint name="my" type="slide" axis="0 1 0" limited="false"/>
      <geom name="target_geom" type="cylinder" size="0.045 0.004"
            rgba="0.13 0.72 0.34 0.35" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <!-- position actuators: force = kp*(target-q) - kv*qvel, so the arm can
         build sustained push force against a load (velocity actuators are
         spring-dampers on velocity with no integral term — they cap out at
         kv*(ctrl-qvel) ~1.4 Nm when stalled, far too weak to push). -->
    <position name="act_j1" joint="j1" kp="50" kv="4"
              ctrlrange="-2.8 2.8" forcerange="-40 40"/>
    <position name="act_j2" joint="j2" kp="50" kv="4"
              ctrlrange="-2.4 2.4" forcerange="-40 40"/>
  </actuator>
</mujoco>
"""


def _build(model_xml: str = MJCF):
    import mujoco

    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)
    return mujoco, model, data


class MuJoCoPusher:
    """Contact-rich 2-link arm pushing a block to a target (MuJoCo).

    Mirrors the PlanarPusher API exactly (reset / step / render / success /
    done / final_dist / trajectory), so policies, the expert, curation
    strategies, and the flywheel loop run unchanged on real contact
    dynamics.
    """

    def __init__(
        self,
        seed: int = 0,
        horizon: int = 70,
        dt: float = DT,
        rng: Optional[np.random.Generator] = None,
        success_radius: float = SUCCESS_RADIUS,
        target_ring: tuple[float, float] = (0.08, 0.18),
        record_images: bool = False,
        img_size: int = IMG_SIZE,
    ):
        self.mujoco, self.model, self.data = _build()
        self.horizon = horizon
        self.dt = dt
        self.success_radius = success_radius
        self.target_ring = target_ring
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.record_images = record_images
        self.img_size = img_size
        self.model.opt.timestep = dt / max(1, int(round(dt / 0.01)))  # ~0.01 substeps
        self._substeps = max(1, int(round(dt / self.model.opt.timestep)))
        self._renderer = None
        if record_images:
            self._make_renderer()
        self.t = 0
        self.obs: Optional[Obs] = None
        self.in_goal_steps = 0
        self._traj_states: list[np.ndarray] = []
        self._traj_actions: list[np.ndarray] = []
        self._traj_images: list[np.ndarray] = []
        self._q_cmd: np.ndarray = np.zeros(2)  # joint position targets (position actuators)

    # ------------------------------------------------------------------ #
    # model internals                                                     #
    # ------------------------------------------------------------------ #
    def _make_renderer(self):
        import mujoco

        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.img_size, width=self.img_size)
        return self._renderer

    def _qpos_adr(self, joint: str) -> int:
        mj = self.mujoco
        return self.model.jnt_qposadr[
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint)]

    def _qvel_adr(self, joint: str) -> int:
        mj = self.mujoco
        return self.model.jnt_dofadr[
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint)]

    def _read_obs(self) -> Obs:
        d = self.data
        q = np.array([d.qpos[self._qpos_adr("j1")],
                      d.qpos[self._qpos_adr("j2")]], dtype=np.float64)
        dq = np.array([d.qvel[self._qvel_adr("j1")],
                       d.qvel[self._qvel_adr("j2")]], dtype=np.float64)
        mj = self.mujoco
        block = d.xpos[mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "block")][:2].astype(np.float64)
        vadr = self._qvel_adr("block_free")
        dblock = np.array([d.qvel[vadr + 0], d.qvel[vadr + 1]], dtype=np.float64)
        tip_pos = d.xpos[mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "tip")]
        tip = np.array([tip_pos[0], tip_pos[1]], dtype=np.float64)
        target = np.array([d.qpos[self._qpos_adr("mx")],
                           d.qpos[self._qpos_adr("my")]], dtype=np.float64)
        return Obs(q=q, dq=dq, block=block, dblock=dblock, target=target, tip=tip)

    # ------------------------------------------------------------------ #
    # PlanarPusher API                                                    #
    # ------------------------------------------------------------------ #
    def render(self, obs: Optional[Obs] = None) -> np.ndarray:
        if self.record_images:
            rend = self._make_renderer()
            rend.update_scene(self.data, camera="top")
            img = rend.render()
            return img.copy()
        # no renderer allocated: synthesize a flat scene placeholder
        return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

    def sample_start(self) -> Obs:
        """Same rejection sampling as PlanarPusher (pushable tasks only)."""
        block = self.rng.uniform([0.38, -0.22], [0.60, 0.22])
        r_min, r_max = self.target_ring
        best = None
        best_out = -np.inf
        for _ in range(500):
            ang = self.rng.uniform(0.0, 2 * np.pi)
            rad = self.rng.uniform(r_min, r_max)
            target = block + rad * np.array([np.cos(ang), np.sin(ang)])
            target[0] = np.clip(target[0], 0.15, 0.75)
            target[1] = np.clip(target[1], -0.30, 0.30)
            push_dir = target - block
            pn = np.linalg.norm(push_dir)
            if pn < 1e-6:
                continue
            outward = np.dot(push_dir / pn, block / max(np.linalg.norm(block), 1e-6))
            if outward > 0.25:
                best, best_out = target, outward
                break
            if outward > best_out:
                best, best_out = target, outward
        target = best if best is not None else target
        push_dir = target - block
        push_dir = push_dir / np.linalg.norm(push_dir)
        # start well outside MuJoCo's contact range (sphere r=0.02 + block
        # r=0.045 = 0.065) so the expert's approach phase completes and the
        # push phase actually engages — a tip that starts inside contact
        # drags the block while chasing its stand-off point, forever.
        tip = block - push_dir * 0.12 + self.rng.normal(0.0, 0.012, size=2)
        q = ik2(tip, elbow="down") + self.rng.normal(0.0, 0.03, size=2)
        q = np.clip(q, [-2.8, -2.4], [2.8, 2.4])
        return Obs(q=q, dq=np.zeros(2), block=block, dblock=np.zeros(2),
                   target=target, tip=forward_kinematics(q))

    def reset(self, obs: Optional[Obs] = None) -> Obs:
        d = self.data
        o = obs if obs is not None else self.sample_start()
        d.qpos[self._qpos_adr("j1")] = o.q[0]
        d.qpos[self._qpos_adr("j2")] = o.q[1]
        d.qvel[self._qvel_adr("j1")] = 0.0
        d.qvel[self._qvel_adr("j2")] = 0.0
        # block: position + identity orientation, zero velocity
        adr = self._qpos_adr("block_free")
        d.qpos[adr + 0] = o.block[0]
        d.qpos[adr + 1] = o.block[1]
        d.qpos[adr + 2] = 0.030
        d.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
        vadr = self._qvel_adr("block_free")
        d.qvel[vadr:vadr + 6] = 0.0
        # target marker
        d.qpos[self._qpos_adr("mx")] = o.target[0]
        d.qpos[self._qpos_adr("my")] = o.target[1]
        self.mujoco.mj_forward(self.model, d)
        self._q_cmd = o.q.copy()
        self.t = 0
        self.in_goal_steps = 0
        self.obs = self._read_obs()
        self._traj_states = [state_vector(self.obs)]
        self._traj_actions = []
        self._traj_images = [self.render()] if self.record_images else []
        return self.obs

    def step(self, action: np.ndarray) -> Obs:
        assert self.obs is not None, "call reset() first"
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        q_dot = a * MAX_Q_DOT
        # integrate the velocity command into joint position targets and clip
        # to the joint ranges (position actuators hold sustained push force)
        self._q_cmd = np.clip(self._q_cmd + q_dot * self.dt, [-2.8, -2.4], [2.8, 2.4])
        d = self.data
        mj = self.mujoco
        d.ctrl[mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "act_j1")] = self._q_cmd[0]
        d.ctrl[mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "act_j2")] = self._q_cmd[1]
        for _ in range(self._substeps):
            self.mujoco.mj_step(self.model, d)
        self.obs = self._read_obs()

        in_goal = np.linalg.norm(self.obs.block - self.obs.target) < self.success_radius
        self.in_goal_steps = self.in_goal_steps + 1 if in_goal else 0
        self.t += 1
        self._traj_states.append(state_vector(self.obs))
        self._traj_actions.append(a.copy())
        if self.record_images:
            self._traj_images.append(self.render())
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
        states = np.stack(self._traj_states)
        actions = np.stack(self._traj_actions) if self._traj_actions else np.zeros((0, 2))
        if len(actions) and len(states) == len(actions) + 1:
            states = states[:-1]
        out = {
            "states": states,
            "actions": actions,
            "success": bool(self.success),
            "final_dist": self.final_dist,
            "steps": self.t,
        }
        if self._traj_images:
            imgs = np.stack(self._traj_images)
            if len(actions) and len(imgs) == len(actions) + 1:
                imgs = imgs[:-1]
            out["images"] = imgs
        return out
