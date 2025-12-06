"""
Figure-level wrapper for validation loss plots.

This module assembles the layout and delegates axis drawing to reusable
core functions. Legends and saving are intentionally left to the caller.
"""
from __future__ import annotations

from typing import List, Optional, Sequence
import matplotlib.pyplot as plt

from plot_styles.core.line_plot import plot_models_line
from plot_styles.core.style_axis import style_axis_basic


DEFAULT_LOSS_STYLE = {
    "xscale": "log",
    "x_label": "Effective steps",
}


def plot_loss(
    loss_df,
    *,
    train_sizes: Sequence[int],
    model_order: Sequence[str],
    dataset_markers: dict,
    dataset_pretty: dict,
    y_label: str = "Val loss",
    y_min: Optional[float] = None,
    fig_size=(12, 6),
    multi_panel_config: Optional[dict] = None,
    grid: bool = False,
    save_path: Optional[str] = None,
) -> tuple:
    """Draw loss curves on one or more axes.

    Parameters
    ----------
    loss_df : DataFrame
        Contains columns ``model``, ``train_size``, ``effective_step``,
        ``val_loss`` and optional ``head``.
    multi_panel_config : dict, optional
        Example: ``{"n_rows": 1, "n_cols": 2, "configs": ["Cls", "Cls+Seg"]}``
        will render separate heads on each axis.
    save_path : str, optional
        If provided, the figure is saved but the function still returns
        the figure/axes for further composition.
    """
    axes = []
    if multi_panel_config is None:
        fig, ax = plt.subplots(figsize=fig_size)
        axes.append(ax)
        panel_heads = [None]
    else:
        fig, axes = plt.subplots(
            multi_panel_config["n_rows"],
            multi_panel_config["n_cols"],
            figsize=fig_size,
            sharex=False,
            sharey=False,
        )
        axes = axes.flatten()
        panel_heads = multi_panel_config.get("configs", [None] * len(axes))

    all_active: List[str] = []
    for ax, head_filter in zip(axes, panel_heads):
        active_models = plot_models_line(
            ax,
            loss_df,
            x_col="effective_step",
            y_col="val_loss",
            model_order=model_order,
            train_sizes=train_sizes,
            dataset_markers=dataset_markers,
            head_filter=head_filter,
        )
        style_axis_basic(
            ax,
            y_label=y_label,
            y_min=y_min,
            grid=grid,
            **DEFAULT_LOSS_STYLE,
        )
        all_active.extend(active_models)

    # preserve input ordering
    ordered_active = [m for m in model_order if m in set(all_active)]

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, axes, ordered_active
