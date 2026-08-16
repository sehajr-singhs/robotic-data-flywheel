"""DQN-from-scratch baseline: classic RL on the same task, no demonstrations.

The label-efficiency claim needs an anchor: how much *environment interaction*
does vanilla RL require versus the flywheel's curated labels? DQN receives
the full kinematic state (never pixels), a dense reward (negative distance to
target, a bonus for holding the block in the goal), a 5x5 discretized action
grid, and a generous interaction budget — and still must discover pushing
from scratch. The flywheel instead gets a handful of expert labels and
relabels its own deployment failures.

Output is a per-evaluation curve of held-out success vs environment steps, so
the paper can plot both on the same interaction axis.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..envs.planar_pusher import PlanarPusher, state_vector

ACTION_GRID = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
N_ACTIONS = len(ACTION_GRID) ** 2  # 25 discrete (dq1, dq2) commands

_REPLAY_CAP = 200_000


def _qnet(hidden: int = 128):
    from torch import nn

    return nn.Sequential(
        nn.Linear(12, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, N_ACTIONS),
    )


def _act_to_vec(a: int) -> np.ndarray:
    d1, d2 = divmod(a, len(ACTION_GRID))
    return np.array([ACTION_GRID[d1], ACTION_GRID[d2]])


def train_dqn(
    env: PlanarPusher,
    eval_starts: list,
    budget: int = 300_000,
    seed: int = 0,
    hidden: int = 128,
    batch_size: int = 128,
    gamma: float = 0.99,
    lr: float = 1e-3,
    target_every: int = 2_000,
    eval_every: int = 20_000,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 100_000,
    verbose: bool = True,
) -> dict:
    """Run DQN for `budget` environment steps; return success-vs-steps curve."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    q = _qnet(hidden)
    qt = _qnet(hidden)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    lossf = nn.MSELoss()

    replay: list[tuple] = []
    rng = np.random.default_rng(seed)
    env.rng = rng
    steps = 0
    ep_returns: list[float] = []   # return of each finished episode
    cur_return = 0.0
    loss_history: list[float] = []
    evals: list[dict] = []

    def _act_greedy(s: np.ndarray) -> int:
        with torch.no_grad():
            return int(torch.argmax(q(torch.from_numpy(s.astype(np.float32))[None])).item())

    def _evaluate() -> float:
        succ = 0
        for start in eval_starts:
            env.reset(start)
            while not env.done:
                env.step(_act_to_vec(_act_greedy(state_vector(env.obs))))
            succ += int(env.success)
        return succ / len(eval_starts)

    while steps < budget:
        env.reset()
        s = state_vector(env.obs)
        while not env.done and steps < budget:
            eps = max(eps_end, eps_start - (eps_start - eps_end) * steps / eps_decay_steps)
            if rng.random() < eps:
                a = int(rng.integers(N_ACTIONS))
            else:
                a = _act_greedy(s)
            env.step(_act_to_vec(a))
            s2 = state_vector(env.obs)
            d = float(np.linalg.norm(env.obs.block - env.obs.target))
            in_goal = d < env.success_radius
            r = -d - 0.02 + (1.0 if in_goal else 0.0) + (5.0 if env.success else 0.0)
            done = bool(env.done)
            cur_return += r
            replay.append((s.copy(), a, r, s2.copy(), done))
            if len(replay) > _REPLAY_CAP:
                replay.pop(0)
            s = s2
            steps += 1
            if done:
                ep_returns.append(cur_return)
                cur_return = 0.0

            if steps % target_every == 0:
                qt.load_state_dict(q.state_dict())

            if len(replay) >= batch_size and steps % 4 == 0:
                idx = rng.integers(0, len(replay), size=batch_size)
                mb = [replay[i] for i in idx]
                S = torch.from_numpy(np.stack([m[0] for m in mb]).astype(np.float32))
                A = torch.tensor([m[1] for m in mb], dtype=torch.long)
                R = torch.tensor([m[2] for m in mb], dtype=torch.float32)
                S2 = torch.from_numpy(np.stack([m[3] for m in mb]).astype(np.float32))
                D = torch.tensor([m[4] for m in mb], dtype=torch.float32)
                with torch.no_grad():
                    # double-Q: online net picks the action, target net prices it
                    # (reduces the max-bias that stalls classic DQN on sparse goals)
                    a2 = q(S2).argmax(dim=1, keepdim=True)
                    target = R + gamma * qt(S2).gather(1, a2).squeeze(1) * (1.0 - D)
                pred = q(S).gather(1, A[:, None]).squeeze(1)
                loss = lossf(pred, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_history.append(float(loss.item()))

            if steps % eval_every == 0:
                sr = _evaluate()
                recent = float(np.mean(ep_returns[-50:])) if ep_returns else 0.0
                evals.append({"env_steps": steps, "success_rate": sr,
                              "mean_return": recent})
                if verbose:
                    print(f"  [dqn] steps {steps:7d}  eval success {sr:.2f}  "
                          f"mean return {recent:+.2f}")

    if not evals or evals[-1]["env_steps"] < steps:
        sr = _evaluate()
        recent = float(np.mean(ep_returns[-50:])) if ep_returns else 0.0
        evals.append({"env_steps": steps, "success_rate": sr, "mean_return": recent})
    return {
        "total_env_steps": steps,
        "eval": evals,
        "final_success": evals[-1]["success_rate"],
        "mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
        "loss_final": float(np.mean(loss_history[-200:])) if loss_history else 0.0,
        "loss_early": float(np.mean(loss_history[:200])) if loss_history else 0.0,
        "config": {"budget": budget, "hidden": hidden, "gamma": gamma, "lr": lr,
                   "actions": N_ACTIONS, "replay_cap": _REPLAY_CAP,
                   "eps_decay_steps": eps_decay_steps},
    }
