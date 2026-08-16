"""Plots: success curves, the flywheel report, and example trajectories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .envs.planar_pusher import L1, L2, SUCCESS_RADIUS, forward_kinematics

COLORS = {
    "none": "#888888",
    "success_only": "#2f6fb3",
    "near_miss": "#d97706",
    "relabel": "#16a34a",
    "relabel_plus_success": "#dc2626",
    "success_coverage": "#7c3aed",
}


def plot_success_curves(summary: dict, path: str | Path) -> None:
    """Success rate vs flywheel iteration, one line per strategy (mean +/- std)."""
    iters = list(range(summary["config"]["iterations"] + 1))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, data in summary["strategies"].items():
        mean = data["success_rate_mean"]
        std = data.get("success_rate_std", [0.0] * len(mean))
        color = COLORS.get(name, "#333333")
        ax.plot(iters, mean, marker="o", lw=2, color=color, label=name.replace("_", " "))
        ax.fill_between(iters, np.array(mean) - np.array(std),
                        np.array(mean) + np.array(std), color=color, alpha=0.15)
    ax.set_xlabel("flywheel iteration")
    ax.set_ylabel("held-out success rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(iters)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_flywheel_report(report: dict, path: str | Path) -> None:
    """The deployment report for one strategy's final iteration."""
    rollouts = report["rollouts"]
    fds = np.array([r["final_dist"] for r in rollouts])
    cov = np.array([r["coverage"] for r in rollouts])
    succ = np.array([r["success"] for r in rollouts])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))

    # 1) final-distance histogram with success threshold
    ax = axes[0]
    ax.hist(fds, bins=16, color="#4b5563", alpha=0.85)
    ax.axvline(SUCCESS_RADIUS, color="#dc2626", ls="--", lw=1.5, label="success radius")
    ax.axvline(0.15, color="#d97706", ls=":", lw=1.5, label="near-miss cutoff")
    ax.set_xlabel("final distance to target")
    ax.set_ylabel("episodes")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("deployment outcomes")

    # 2) outcome buckets
    ax = axes[1]
    n_succ = int(succ.sum())
    n_near = int(((~succ) & (fds < 0.15)).sum())
    n_far = len(rollouts) - n_succ - n_near
    ax.bar(["success", "near-miss", "far fail"], [n_succ, n_near, n_far],
           color=["#16a34a", "#d97706", "#888888"])
    ax.set_title("what the flywheel collected")
    for i, v in enumerate([n_succ, n_near, n_far]):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=9)

    # 3) coverage vs outcome (novelty signal)
    ax = axes[2]
    ax.scatter(cov[succ], fds[succ], s=18, color="#16a34a", alpha=0.8, label="success")
    ax.scatter(cov[~succ], fds[~succ], s=18, color="#dc2626", alpha=0.6, label="fail")
    ax.set_xlabel("state coverage (novelty)")
    ax.set_ylabel("final distance")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("coverage vs outcome")

    fig.suptitle(f"flywheel report — iteration {report['iteration']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_budget_comparison(
    summary: dict,
    dqn: dict,
    path: str | Path,
    strategies: tuple[str, ...] = ("success_only", "relabel_curated", "relabel"),
) -> None:
    """Held-out success vs *environment interaction* for the flywheel vs DQN.

    The x-axis is the one axis that matters for label efficiency: total
    environment steps consumed for training data. The flywheel's interaction
    at each iteration is the dataset frame count (seed demos + curated
    rollouts, from the curation log); DQN reports its own step budget.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for name in strategies:
        d = summary["strategies"][name]
        log = d.get("curation_log", [])
        if not log:
            continue
        # training interaction grows with the dataset; eval is measurement only
        inter = [d["success_rate_mean"][0]] + [c["dataset_frames"] for c in log]
        succ = d["success_rate_mean"]
        color = COLORS.get(name, "#333333")
        ax.plot(inter, succ, marker="o", lw=2, color=color,
                label=f"flywheel · {name.replace('_', ' ')}")

    if dqn:
        steps = [e["env_steps"] for e in dqn["eval"]]
        succ = [e["success_rate"] for e in dqn["eval"]]
        ax.plot(steps, succ, marker="s", lw=2, color="#dc2626",
                label="DQN from scratch (state, dense reward)")

    ax.set_xscale("log")
    ax.set_xlabel("environment interactions used for training")
    ax.set_ylabel("held-out success rate")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_vision_curves(summary: dict, path: str | Path) -> None:
    """Same success-curve plot, for a vision (pixel-observation) summary."""
    plot_success_curves(summary, path)


def plot_sample_images(env, starts: list, path: str | Path, n: int = 4) -> None:
    """A montage of raw pixel observations — what the vision policy sees."""
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    for ax, start in zip(axes, starts[:n]):
        ax.imshow(env.render(start))
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _draw_arm(ax, q: np.ndarray, **kw) -> None:
    elbow = L1 * np.array([np.cos(q[0]), np.sin(q[0])])
    tip = forward_kinematics(q)
    ax.plot([0, elbow[0]], [0, elbow[1]], color="#1f2937", lw=2, **kw)
    ax.plot([elbow[0], tip[0]], [elbow[1], tip[1]], color="#374151", lw=2, **kw)
    ax.plot(*elbow, "o", ms=4, color="#1f2937")
    ax.plot(*tip, "o", ms=4, color="#dc2626")


def plot_trajectories(
    trajs: dict[str, dict],
    path: str | Path,
) -> None:
    """One panel per trajectory: arm at key frames, block path, target."""
    names = list(trajs)
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 3.6), sharey=True)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        t = trajs[name]
        states = np.asarray(t["states"])
        block_path = states[:, 4:6]
        target = states[0, 8:10]
        for frac in np.linspace(0.0, 1.0, 5):
            i = min(len(states) - 1, int(frac * (len(states) - 1)))
            _draw_arm(ax, states[i, 0:2], alpha=0.35)
        ax.plot(block_path[:, 0], block_path[:, 1], color="#16a34a", lw=1.5,
                label="block path")
        ax.plot(*block_path[0], "o", color="#16a34a", ms=6)
        ax.plot(*block_path[-1], "s", color="#059669", ms=6)
        circle = plt.Circle(target, SUCCESS_RADIUS, color="#dc2626", alpha=0.25)
        ax.add_patch(circle)
        ax.plot(*target, "+", color="#dc2626", ms=10)
        ax.set_title(f"{name}\nsuccess={t['success']}, d={t['final_dist']:.3f}", fontsize=9)
        ax.set_xlim(-0.15, 0.95)
        ax.set_ylim(-0.55, 0.55)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    axes[0].set_xlabel("x (m)")
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(path, dpi=160)
    plt.close(fig)
