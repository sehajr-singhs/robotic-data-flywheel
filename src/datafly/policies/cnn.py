"""Vision policy: pixels -> joint-velocity commands, trained by behavior cloning.

This is the *perception-grounded* counterpart to the numpy MLP: instead of
the 12-dim kinematic state vector, the policy consumes the raw rendered image
and must infer where the block and target are. Torch is imported lazily so the
package remains usable without it (the state-based study has zero heavy
dependencies); the vision study runs where torch + GPU are available.

The API mirrors MLPPolicy exactly (`act(obs) -> action`), so the flywheel
loop is policy-agnostic: swap `obs_mode="state"` for `obs_mode="image"` and
the same collection -> curate -> retrain loop runs.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..envs.planar_pusher import ACTION_DIM, IMG_SIZE


class CNNPolicy:
    """Small conv net: 64x64 RGB -> 3 conv blocks -> MLP -> 2 (tanh)."""

    def __init__(self, img_size: int = IMG_SIZE, hidden: int = 128, seed: int = 0):
        import torch

        torch.manual_seed(seed)
        self.img_size = img_size
        self.hidden = hidden
        self.net = self._build(img_size, hidden)

    @staticmethod
    def _build(img_size: int, hidden: int):
        from torch import nn

        conv_dim = img_size // 8  # three stride-2 convs
        return nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * conv_dim * conv_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, ACTION_DIM),
            nn.Tanh(),
        )

    def _tensor(self, X: np.ndarray):
        """(B, H, W, 3) uint8/float -> (B, 3, H, W) float tensor in [0, 1]."""
        import torch

        x = np.asarray(X, dtype=np.float32)
        x = x.reshape(-1, self.img_size, self.img_size, 3) / 255.0
        return torch.from_numpy(x).permute(0, 3, 1, 2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """(B, H, W, 3) images -> (B, 2) actions."""
        with self.net_ctx():
            return self.net(self._tensor(X)).numpy()

    def act(self, img: np.ndarray) -> np.ndarray:
        return self.predict(np.asarray(img).reshape(1, self.img_size, self.img_size, 3))[0]

    def net_ctx(self):
        import torch

        return torch.no_grad()


def train_bc_image(
    trjs: list,
    hidden: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 0,
    init: Optional[CNNPolicy] = None,
    img_size: int = IMG_SIZE,
    verbose: bool = False,
) -> CNNPolicy:
    """Behavior clone a CNN from (image, action) pairs; `init` fine-tunes.

    `init` continues from an existing policy's weights (flywheel fine-tuning),
    matching the state-based `train_bc` contract.
    """
    import torch
    from torch import nn

    imgs = np.concatenate([t.images for t in trjs], axis=0)
    acts = np.concatenate([t.actions for t in trjs], axis=0).astype(np.float32)
    if init is not None:
        policy = CNNPolicy(img_size=img_size, hidden=init.hidden, seed=seed)
        policy.net.load_state_dict(init.net.state_dict())
    else:
        policy = CNNPolicy(img_size=img_size, hidden=hidden, seed=seed)

    n = len(imgs)
    if n == 0:
        return policy

    X = policy._tensor(imgs)
    y = torch.from_numpy(acts)
    opt = torch.optim.Adam(policy.net.parameters(), lr=lr)
    lossf = nn.MSELoss()
    rng = np.random.default_rng(seed + 1)

    for ep in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss = lossf(policy.net(X[idx]), y[idx])
            loss.backward()
            opt.step()
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  [cnn] epoch {ep:4d}  loss {loss.item():.5f}  ({n} frames)")

    return policy
