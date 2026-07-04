"""
Visualisations for forehand tennis stroke analysis.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR

# Consistent colour palette across both plots
_LABEL_COLORS: dict[str, str] = {
    "Flat Drive": "#2ecc71",
    "Standard Topspin": "#3498db",
    "Heavy Topspin": "#e74c3c",
    "Unknown": "#95a5a6",
}


def plot_swing_path_overlay(
    frame: np.ndarray,
    racket_positions: list[dict | None],
    contact_frame: int,
    lookback: int = 3,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Draw the swing path as an arrow on the contact-frame image.

    The arrow runs from the racket-head centre *lookback* frames before contact
    to the contact-frame racket-head centre.  When racket positions are
    unavailable (detector not yet implemented), a placeholder message is shown.

    Saves to ``reports/figures/`` when *save_path* is not provided.

    NOTE: Overlay is a 2D projection — does not represent true 3D swing direction.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Swing Path Overlay — contact frame {contact_frame}", fontsize=13)
    ax.axis("off")

    pre_idx = max(0, contact_frame - lookback)
    pre = racket_positions[pre_idx]
    contact = racket_positions[contact_frame]

    if pre is not None and contact is not None:
        px = pre["bbox"][0] + pre["bbox"][2] / 2
        py = pre["bbox"][1] + pre["bbox"][3] / 2
        cx = contact["bbox"][0] + contact["bbox"][2] / 2
        cy = contact["bbox"][1] + contact["bbox"][3] / 2

        ax.annotate(
            "",
            xy=(cx, cy),
            xytext=(px, py),
            arrowprops=dict(arrowstyle="->", color="yellow", lw=2.5),
        )
        ax.plot(px, py, "go", ms=8, label=f"Pre-contact (frame {pre_idx})")
        ax.plot(cx, cy, "ro", ms=8, label=f"Contact (frame {contact_frame})")
        ax.legend(loc="upper left", fontsize=9)
    else:
        ax.text(
            0.5,
            0.5,
            "Racket positions unavailable\n(detector not yet implemented — see dataset.localize_racket)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            color="red",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8),
        )

    if save_path is None:
        save_path = FIGURES_DIR / f"swing_path_frame{contact_frame}.png"
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_angle_distribution(
    df: pd.DataFrame,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Scatter plot of racket face angle vs. swing path angle, coloured by shot label.

    Expects a DataFrame with columns ``swing_path_angle_deg``,
    ``racket_face_angle_deg``, and ``shot_label`` (as produced by
    ``features.compute_features``).  Rows with NaN in either angle column are
    dropped from the plot.

    Saves to ``reports/figures/`` when *save_path* is not provided.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    plot_df = df.dropna(subset=["swing_path_angle_deg", "racket_face_angle_deg"])

    if plot_df.empty:
        ax.text(
            0.5,
            0.5,
            "No data to display.\nRun the full pipeline with real video input first.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            color="gray",
        )
    else:
        for label, group in plot_df.groupby("shot_label"):
            color = _LABEL_COLORS.get(str(label), "#7f8c8d")
            ax.scatter(
                group["swing_path_angle_deg"],
                group["racket_face_angle_deg"],
                label=label,
                color=color,
                alpha=0.75,
                edgecolors="k",
                linewidths=0.5,
                s=80,
            )

    ax.set_xlabel("Swing Path Angle (°)", fontsize=12)
    ax.set_ylabel("Racket Face Angle (°)", fontsize=12)
    ax.set_title("Forehand Angle Distribution by Shot Type", fontsize=13)
    ax.legend(title="Shot Label", fontsize=9)
    ax.grid(True, alpha=0.3)

    if save_path is None:
        save_path = FIGURES_DIR / "angle_distribution.png"
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
