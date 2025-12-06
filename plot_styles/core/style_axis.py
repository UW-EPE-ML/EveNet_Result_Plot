"""
Axis styling utilities for modular plot construction.

These helpers avoid figure-level side effects so they can be reused
across composite layouts (multi-panel, inset axes, shared legends).
"""
from __future__ import annotations

from typing import Optional
import os
import matplotlib.pyplot as plt

from plot_styles.core.theme import PlotStyle


def apply_nature_axis_style(ax, *, style: PlotStyle | None = None):
    """Apply a consistent, clean axis style.

    This is adapted from the previous utilities but kept isolated so
    it can be reused by any axis-level plotter.
    """
    scale = style.object_scale if style is not None else 1.0
    tick_size = (
        style.tick_label_size
        if style is not None and style.tick_label_size is not None
        else plt.rcParams.get("xtick.labelsize", 11)
    )

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
        length=5 * scale,
        width=1.0 * scale,
        labelsize=tick_size,
        pad=3 * scale,
        bottom=True,
        top=False,
        left=True,
        right=False,
    )

    label_pad = 4 * scale
    ax.xaxis.labelpad = label_pad
    ax.yaxis.labelpad = label_pad


def style_axis_basic(
        ax,
        *,
        xscale: Optional[str] = None,
        yscale: Optional[str] = None,
        y_label: str = "",
        x_label: str = "",
        y_max: Optional[float] = None,
        y_min: Optional[float] = None,
        x_indicator: Optional[float] = None,
        x_indicator_text_config: Optional[dict] = None,
        y_indicator: Optional[float] = None,
        y_indicator_text_config: Optional[dict] = None,
        grid: bool = False,
        style: PlotStyle | None = None,
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
    if y_max is not None:
        ax.set_ylim(top=y_max)

    apply_nature_axis_style(ax, style=style)

    if grid:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    if x_indicator is not None:
        ax.axvline(x=x_indicator, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        if x_indicator_text_config:
            ax.text(
                x_indicator * x_indicator_text_config.get("fraction_x", 0.95),
                x_indicator_text_config.get("fraction_y", 0.025),
                x_indicator_text_config.get("fmt", "Typical HEP dataset: 100k events"),
                fontsize=x_indicator_text_config.get("fontsize", 12),
                color=x_indicator_text_config.get("color", "gray"),
                ha=x_indicator_text_config.get("ha", "right"),
            )
    if y_indicator is not None:
        ax.axhline(y=y_indicator, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        if y_indicator_text_config:
            ax.text(
                y_indicator_text_config.get("fraction_x", 0.025),
                y_indicator * y_indicator_text_config.get("fraction_y", 0.95),
                y_indicator_text_config.get("fmt", "Typical HEP dataset: 100k events"),
                fontsize=y_indicator_text_config.get("fontsize", 12),
                color=y_indicator_text_config.get("color", "gray"),
                ha=y_indicator_text_config.get("ha", "right"),
            )


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
