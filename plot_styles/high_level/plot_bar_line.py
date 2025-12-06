"""
Two-panel bar/line wrapper built on reusable core plotters.
"""
from __future__ import annotations

from typing import Optional, Sequence
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from plot_styles.core.bar_plot import plot_model_bars
from plot_styles.core.line_plot import plot_models_line
from plot_styles.core.style_axis import style_axis_basic
from plot_styles.core.theme import PlotStyle, scaled_fig_size, use_style


def plot_bar_line(
    data_df,
    *,
    metric: str,
    model_order: Sequence[str],
    train_sizes: Sequence[int],
    dataset_markers: dict,
    dataset_pretty: dict,
    head_order: Optional[Sequence[str]] = None,
    y_label: str = "Metric",
    x_label: str = "Train Size [K]",
    y_min: Optional[float] = None,
    fig_size=(14, 6),
    fig_scale: float = 1.0,
    fig_aspect: Optional[float] = None,
    panel_ratio=(3, 2),
    x_indicator: Optional[float] = None,
    logy: bool = False,
    logx: bool = False,
    save_path: Optional[str] = None,
    bar_train_size: Optional[int] = None,
    style: PlotStyle | None = None,
) -> tuple:
    """Render paired bar/line panels for a given metric.

    Returns the figure, axes tuple and the ordered list of active models.
    """
    if head_order is None:
        head_order = [None]

    with use_style(style):
        resolved_size = scaled_fig_size(fig_size, scale=fig_scale, aspect_ratio=fig_aspect)
        fig = plt.figure(figsize=resolved_size)
        gs = GridSpec(1, 2, width_ratios=panel_ratio)
        ax_left = fig.add_subplot(gs[0])
        ax_right = fig.add_subplot(gs[1])

    train_size_for_bar = bar_train_size or max(train_sizes)
    active_bars = plot_model_bars(
        ax_left,
        data_df,
        metric=metric,
        model_order=model_order,
        head_order=head_order,
        train_size_for_bar=train_size_for_bar,
        bar_kwargs={"linewidth": 0.8 * (style.object_scale if style else 1.0)},
    )

    active_lines = plot_models_line(
        ax_right,
        data_df,
        x_col="train_size",
        y_col=metric,
        model_order=model_order,
        train_sizes=train_sizes,
        dataset_markers=dataset_markers,
        head_order=head_order if head_order != [None] else None,
        size_scale=style.object_scale if style else 1.0,
    )

    line_scale = "log" if logx else None
    y_scale = "log" if logy else None

    style_axis_basic(
        ax_left,
        y_label=y_label,
        x_label="Full dataset size" if head_order == [None] else "Head",
        y_min=y_min[0] if isinstance(y_min, (tuple, list)) else y_min,
        yscale=y_scale,
        style=style,
    )
    style_axis_basic(
        ax_right,
        y_label=y_label,
        x_label=x_label,
        y_min=y_min[1] if isinstance(y_min, (tuple, list)) else y_min,
        xscale=line_scale,
        yscale=y_scale,
        x_indicator=x_indicator,
        style=style,
    )

    active_models = [m for m in model_order if m in set(active_bars + active_lines)]

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, (ax_left, ax_right), active_models
