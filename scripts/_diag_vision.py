"""Lean capacity probe: small vs current CNN under blind relabeling.

Usage: python scripts/_diag_vision.py small|current
"""

import sys

import numpy as np

from datafly.envs.planar_pusher import PlanarPusher
from datafly.eval import evaluate, make_eval_starts
from datafly.policies.cnn import CNNPolicy, train_bc_image
from datafly.policies.expert import ScriptedExpert
from datafly.policies.mlp import collect_trajectories
from datafly.curation.strategies import relabel

IMG = 24
EPOCHS = 12
FT = 6
ITERS = 2

env = PlanarPusher(seed=0, horizon=70, record_images=True, img_size=IMG)
expert = ScriptedExpert(noise=0.12, rng=np.random.default_rng(7))
oracle = ScriptedExpert(noise=0.0, rng=np.random.default_rng(17))
eval_starts = make_eval_starts(env, 100, seed=2)


class SmallCNN(CNNPolicy):
    @staticmethod
    def _build(img_size: int, hidden: int):
        from torch import nn

        conv_dim = img_size // 8
        return nn.Sequential(
            nn.Conv2d(3, 4, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(4, 8, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(8, 8, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * conv_dim * conv_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 2), nn.Tanh(),
        )


def run_cnn(label, policy_cls, hidden):
    rng0 = np.random.default_rng(100)
    demos = collect_trajectories(env, None, n=25, rng=rng0,
                                 act_fn=expert.act_from_state, source="expert")
    dataset = list(demos)
    pol = train_bc_image(dataset, hidden=hidden, epochs=EPOCHS, seed=0, img_size=IMG,
                         policy_cls=policy_cls)
    curve = [evaluate(env, pol, eval_starts, obs_fn=env.render)["success_rate"]]
    for it in range(1, ITERS + 1):
        roll_rng = np.random.default_rng(1000 + it)
        rollouts = collect_trajectories(env, pol, n=15, rng=roll_rng,
                                        act_fn=pol.act, source="policy",
                                        seed_base=it * 1000, obs_fn=env.render)
        dataset.extend(relabel(rollouts, expert=oracle))
        pol = train_bc_image(dataset, hidden=hidden, epochs=FT, seed=0,
                             init=pol, img_size=IMG, policy_cls=policy_cls)
        curve.append(evaluate(env, pol, eval_starts, obs_fn=env.render)["success_rate"])
    print(f"{label:34s} curve={[round(float(c), 3) for c in curve]} "
          f"frames={sum(len(t) for t in dataset)}", flush=True)


which = sys.argv[1] if len(sys.argv) > 1 else "both"
if which in ("small", "both"):
    run_cnn("S2  CNN small (4-8-8) relabel", SmallCNN, 48)
if which in ("current", "both"):
    run_cnn("S3  CNN current (16-32-32) relabel", CNNPolicy, 48)
