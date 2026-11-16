"""Viewer: draws the arena, the body and active contacts; saves a short GIF.

Rendering works from recorded arrays (the ``signals.npz`` of a run), never from
the live environment, so drawing cannot perturb a simulation or its timing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

COLORS = {
    "background": "#f7f5f0", "wall": "#3a3a3a", "obstacle": "#7a7a7a", "food": "#3d9a50",
    "neutral": "#c9a66b", "trunk": "#3b6fb6", "limb": "#1f3f73", "contact": "#d43d3d",
}
N_CONTOUR = 8


def contact_points(trunk_pose: np.ndarray, limb_points: np.ndarray, trunk_radius: float) -> np.ndarray:
    """World positions of the 10 touch sensors: 8 contour points, then the 2 tips."""
    x, y, h = trunk_pose
    ang = h + np.arange(N_CONTOUR) * 2 * np.pi / N_CONTOUR
    contour = np.stack([x + trunk_radius * np.cos(ang), y + trunk_radius * np.sin(ang)], axis=1)
    return np.vstack([contour, limb_points[:, 2]])


def save_gif(signals: Mapping[str, np.ndarray], out_path: str | Path, steps: int, stride: int = 2,
             fps: int = 15, title: str = "") -> Path:
    """Render the first ``steps`` recorded steps (every ``stride``-th) to a GIF.

    ``signals`` needs: ``arena_size``, ``trunk_radius``, ``obstacle_pos``,
    ``obstacle_r``, ``food_r``, ``neutral_r`` and per step ``trunk_pose``,
    ``limb_points``, ``food_pos``, ``food_active``, ``neutral_pos``,
    ``touch_intensity``, ``energy``.
    """
    out_path = Path(out_path)
    size = float(signals["arena_size"])
    R = float(signals["trunk_radius"])
    n = min(int(steps), len(signals["trunk_pose"]))
    frames = range(0, n, max(1, int(stride)))

    fig, ax = plt.subplots(figsize=(4, 4.3), dpi=72)
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])
    ax.set_xlim(-0.2, size + 0.2)
    ax.set_ylim(-0.2, size + 0.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), size, size, fill=False, lw=2.5, ec=COLORS["wall"]))
    for p, r in zip(signals["obstacle_pos"], signals["obstacle_r"]):
        ax.add_patch(Circle(p, r, color=COLORS["obstacle"]))
    food = [ax.add_patch(Circle((0, 0), r, color=COLORS["food"])) for r in signals["food_r"]]
    neutral = [ax.add_patch(Circle((0, 0), r, color=COLORS["neutral"], ec="#8a6d3b", lw=1))
               for r in signals["neutral_r"]]
    trunk = ax.add_patch(Circle((0, 0), R, color=COLORS["trunk"], zorder=3))
    (heading_line,) = ax.plot([], [], color="white", lw=2, zorder=4)
    limbs = [ax.plot([], [], color=COLORS["limb"], lw=3, solid_capstyle="round", zorder=5)[0] for _ in range(2)]
    contacts = ax.scatter([], [], s=30, c=COLORS["contact"], zorder=6)
    label = ax.set_title("", fontsize=9)

    def draw(t: int):
        x, y, h = signals["trunk_pose"][t]
        trunk.center = (x, y)
        heading_line.set_data([x, x + R * np.cos(h)], [y, y + R * np.sin(h)])
        for i, line in enumerate(limbs):
            pts = signals["limb_points"][t, i]
            line.set_data(pts[:, 0], pts[:, 1])
        for j, c in enumerate(food):
            c.center = signals["food_pos"][t, j]
            c.set_visible(bool(signals["food_active"][t, j]))
        for j, c in enumerate(neutral):
            c.center = signals["neutral_pos"][t, j]
        pts = contact_points(signals["trunk_pose"][t], signals["limb_points"][t], R)
        active = signals["touch_intensity"][t] > 0
        contacts.set_offsets(pts[active] if active.any() else np.zeros((0, 2)))
        label.set_text(f"{title}  step {t + 1}  energy {signals['energy'][t]:.2f}".strip())
        return []

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path
