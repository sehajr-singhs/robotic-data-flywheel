"""Behavior-cloning policy: a small MLP that maps state -> joint-velocity
commands, plus the training and rollout-collection machinery.

The policy is a plain numpy MLP (manual backprop, mini-batch Adam). This is
deliberate: the whole point of the flywheel study is that *data* — not model
scale or framework — drives improvement, so the policy is a straw-man that
any deployment pipeline could swap for a larger vision-language-action model.
Writing it in numpy keeps the study dependency-free and fast on a laptop:
the full experiment (six curation strategies x four flywheel iterations)
runs in a few minutes with no GPU.

The policy API is exactly what a swap needs: `act(state) -> action`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np

from ..envs.planar_pusher import ACTION_DIM, PlanarPusher, STATE_DIM, state_vector


@dataclass
class Trajectory:
    """One collected rollout (deployment episode)."""
    states: np.ndarray          # (T, STATE_DIM)
    actions: np.ndarray         # (T, ACTION_DIM)
    success: bool
    final_dist: float
    steps: int
    seed: int = 0
    source: str = "policy"      # "expert" | "policy" | "relabeled"
    images: Optional[np.ndarray] = None  # (T, H, W, 3) uint8, if recorded

    def __len__(self) -> int:
        return len(self.states)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


class MLPPolicy:
    """Two-hidden-layer ReLU MLP with tanh output, trained by BC.

    Pure numpy, manual backprop. Shapes: 12 -> hidden -> hidden -> 2.
    """

    def __init__(self, hidden: int = 128, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.hidden = hidden
        # He initialization
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / STATE_DIM), (hidden, STATE_DIM))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / hidden), (hidden, hidden))
        self.b2 = np.zeros(hidden)
        self.W3 = rng.normal(0.0, np.sqrt(2.0 / hidden), (ACTION_DIM, hidden))
        self.b3 = np.zeros(ACTION_DIM)
        # Feature normalization (fit on the training set in train_bc)
        self.mean = np.zeros(STATE_DIM)
        self.std = np.ones(STATE_DIM)

    # ------------------------------------------------------------------ #
    # forward                                                             #
    # ------------------------------------------------------------------ #
    def _forward(self, X: np.ndarray):
        """Return (a3, cache) where a3 = tanh(W3 relu(W2 relu(W1 X)))."""
        X = (X - self.mean) / self.std
        z1 = X @ self.W1.T + self.b1
        a1 = _relu(z1)
        z2 = a1 @ self.W2.T + self.b2
        a2 = _relu(z2)
        z3 = a2 @ self.W3.T + self.b3
        a3 = np.tanh(z3)
        return a3, (X, z1, a1, z2, a2, z3)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)[0]

    def act(self, s: np.ndarray) -> np.ndarray:
        return self.predict(np.asarray(s, dtype=np.float64).reshape(1, -1))[0]

    # ------------------------------------------------------------------ #
    # training                                                            #
    # ------------------------------------------------------------------ #
    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def _backward(self, X, y, cache):
        """Gradients of mean-squared-error loss w.r.t. all parameters."""
        X, z1, a1, z2, a2, z3 = cache
        n = X.shape[0]
        a3 = np.tanh(z3)

        dz3 = 2.0 * (a3 - y) / n
        dW3 = dz3.T @ a2
        db3 = dz3.sum(axis=0)

        da2 = dz3 @ self.W3
        dz2 = da2 * (z2 > 0.0)
        dW2 = dz2.T @ a1
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.W2
        dz1 = da1 * (z1 > 0.0)
        dW1 = dz1.T @ X
        db1 = dz1.sum(axis=0)

        return [dW1, db1, dW2, db2, dW3, db3]

    def _mse(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean((self.predict(X) - y) ** 2))


def train_bc(
    trjs: list[Trajectory],
    hidden: int = 64,
    epochs: int = 120,
    lr: float = 1e-3,
    batch_size: int = 512,
    seed: int = 0,
    init: Optional[MLPPolicy] = None,
    verbose: bool = False,
) -> MLPPolicy:
    """Train an MLP policy by behavior cloning with mini-batch Adam.

    `init` continues training from an existing policy's weights (flywheel
    fine-tuning) instead of starting from scratch each iteration.
    """
    states = np.concatenate([t.states for t in trjs], axis=0).astype(np.float64)
    actions = np.concatenate([t.actions for t in trjs], axis=0).astype(np.float64)
    if init is not None:
        policy = MLPPolicy(hidden=init.hidden, seed=seed)
        policy.W1 = init.W1.copy(); policy.b1 = init.b1.copy()
        policy.W2 = init.W2.copy(); policy.b2 = init.b2.copy()
        policy.W3 = init.W3.copy(); policy.b3 = init.b3.copy()
    else:
        policy = MLPPolicy(hidden=hidden, seed=seed)

    # Fit feature normalization on this training set (carried through fine-tuning).
    policy.mean = states.mean(axis=0)
    policy.std = states.std(axis=0) + 1e-6

    n = len(states)
    if n == 0:
        return policy

    # Adam accumulators
    ms = [np.zeros_like(p) for p in policy.parameters()]
    vs = [np.zeros_like(p) for p in policy.parameters()]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    rng = np.random.default_rng(seed + 1)
    t = 0

    for ep in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            Xb, yb = states[idx], actions[idx]
            t += 1

            cache = policy._forward(Xb)[1]
            grads = policy._backward(Xb, yb, cache)

            for p, g, m, v in zip(policy.parameters(), grads, ms, vs):
                m[:] = beta1 * m + (1 - beta1) * g
                v[:] = beta2 * v + (1 - beta2) * g * g
                mhat = m / (1 - beta1**t)
                vhat = v / (1 - beta2**t)
                p -= lr * mhat / (np.sqrt(vhat) + eps)

        if verbose and (ep % 25 == 0 or ep == epochs - 1):
            print(f"  [bc] epoch {ep:4d}  loss {policy._mse(states, actions):.5f}  ({n} frames)")

    return policy


def collect_trajectories(
    env: PlanarPusher,
    policy,
    n: int,
    rng: Optional[np.random.Generator] = None,
    act_fn: Optional[Callable] = None,
    source: str = "policy",
    seed_base: int = 0,
    horizon: Optional[int] = None,
    obs_fn: Optional[Callable] = None,
) -> list[Trajectory]:
    """Roll out a policy (or any act_fn) from n fresh random starts.

    Each rollout is a standalone episode in the environment; the collection
    RNG is separate from the training RNG so the flywheel loop is
    reproducible iteration by iteration.

    `obs_fn` maps an observation to what the policy consumes. Defaults to
    the state vector; vision policies pass `env.render` (pixels).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if act_fn is None:
        act_fn = policy.act
    if obs_fn is None:
        obs_fn = state_vector
    horizon = horizon or env.horizon

    # The passed rng owns *start sampling*: the environment's own rng may
    # have been advanced by earlier phases (eval sets, prior rollouts), so
    # borrowing it would make this collection depend on call history.
    env.rng = rng

    trjs: list[Trajectory] = []
    for i in range(n):
        env.reset()
        while not env.done:
            env.step(act_fn(obs_fn(env.obs)))
        traj = env.trajectory()
        traj["seed"] = seed_base + i
        traj["source"] = source
        trjs.append(Trajectory(**traj))
    return trjs
