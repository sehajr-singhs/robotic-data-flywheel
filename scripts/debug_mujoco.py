"""Debug the MuJoCo contact-rich port: joint/FK consistency, physics sanity,
and the scripted expert's solve rate (the gate before shipping a kernel).

Usage: python scripts/debug_mujoco.py [--n 100] [--render] [--plot]
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datafly.envs.mujoco_pusher import MuJoCoPusher
    from datafly.envs.planar_pusher import forward_kinematics
    from datafly.policies.expert import PushCommitExpert

    env = MuJoCoPusher(seed=args.seed, record_images=args.render)
    expert = PushCommitExpert(noise=0.0, rng=np.random.default_rng(7), cap_radius=0.07)

    # 1) joint -> FK consistency
    env.reset()
    errs = []
    for _ in range(50):
        o = env.sample_start()
        env.reset(o)
        fk = forward_kinematics(o.q)
        errs.append(np.linalg.norm(fk - o.tip))
    print(f"FK vs sampled tip error: mean {np.mean(errs):.5f} m")

    # 2) expert solve rate
    n_ok = 0
    dists = []
    steps_l = []
    for i in range(args.n):
        o = env.sample_start()
        env.reset(o)
        while not env.done:
            s = env.obs
            from datafly.envs.planar_pusher import state_vector
            a = expert.act_from_state(state_vector(s))
            env.step(a)
        n_ok += int(env.success)
        dists.append(env.final_dist)
        steps_l.append(env.t)
    print(f"expert solve rate: {n_ok}/{args.n} = {n_ok / args.n:.2%}")
    print(f"final dist: mean {np.mean(dists):.3f} m, median {np.median(dists):.3f}, "
          f"max {np.max(dists):.3f}")
    print(f"episode steps: mean {np.mean(steps_l):.0f} (horizon {env.horizon})")

    if args.render:
        from datafly.envs.planar_pusher import render_obs
        o = env.sample_start()
        env.reset(o)
        img = env.render()
        print("render shape:", img.shape, img.dtype, "unique:", np.unique(img).size)


if __name__ == "__main__":
    main()
