"""
Axis styling utilities for modular plot construction.

These helpers avoid figure-level side effects so they can be reused
across composite layouts (multi-panel, inset axes, shared legends).
"""
from __future__ import annotations

from typing import Optional
import os
import matplotlib.pyplot as plt


def apply_nature_axis_style(ax):
    """Apply a consistent, clean axis style.

    This is adapted from the previous utilities but kept isolated so
    it can be reused by any axis-level plotter.
    """
    ax.tick_params(reset=True)

    # Remove grid lines and hide top/right spines
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Emphasize left/bottom spines
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color("black")
    ax.spines['bottom'].set_color("black")

    # Ticks: outward facing, modest sizing
    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        length=5,
        width=1.0,
        labelsize=11,
        pad=3,
        bottom=True,
        top=False,
        left=True,
        right=False,
    )

    ax.xaxis.labelpad = 4
    ax.yaxis.labelpad = 4


def style_axis_basic(
    ax,
    *,
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    y_label: str = "",
    x_label: str = "",
    y_min: Optional[float] = None,
    x_indicator: Optional[float] = None,
    grid: bool = False,
):
    """Apply common axis styling without touching figure layout.

    Parameters
    ----------
    ax : matplotlib axis
    xscale, yscale : {"log", "linear", None}
        Axis scale directives. When ``None`` the current scale is kept.
    y_label, x_label : str
        Labels applied to the respective axes.
    y_min : float, optional
        If provided, enforces a lower y-limit.
    x_indicator : float, optional
        Draws a vertical indicator line at this x-value.
    grid : bool
        Toggle a light grid on both axes.
    """
    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if y_min is not None:
        ax.set_ylim(bottom=y_min)

    apply_nature_axis_style(ax)

    if grid:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    if x_indicator is not None:
        ax.axvline(x=x_indicator, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)


def save_axis(ax, plot_dir: str, f_name: str, dpi: Optional[int] = None):
    """Persist only the provided axis to disk without other panels.

    This temporarily hides sibling axes so the exported PDF/PNG contains
    a single panel. It intentionally avoids modifying the caller's layout
    beyond visibility toggles.
    """
    fig = ax.figure
    original_vis = [a.get_visible() for a in fig.axes]
    for a in fig.axes:
        a.set_visible(False)

    ax.set_visible(True)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    os.makedirs(plot_dir, exist_ok=True)
    plot_des = os.path.join(plot_dir, f_name)
    fig.savefig(plot_des, bbox_inches="tight", dpi=dpi)
    print(f"Saved figure → {plot_des}")

    for a, vis in zip(fig.axes, original_vis):
        a.set_visible(vis)
