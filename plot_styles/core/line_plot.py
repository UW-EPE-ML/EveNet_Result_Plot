"""
Reusable line + scatter plotting utilities for paper figures.

The goal is to keep axis-level drawing logic reusable across
multi-panel layouts while leaving legend and figure management to the
caller.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence
import numpy as np
import pandas as pd

from plot_styles.style import MODEL_COLORS, HEAD_LINESTYLES

DEFAULT_LINE_KWARGS = {
    "linewidth": 3,
    "alpha": 1.0,
}

DEFAULT_SCATTER_KWARGS = {
    "s": 140,
    "edgecolor": "black",
    "linewidth": 1.0,
    "alpha": 0.85,
}


def _select_heads(df: pd.DataFrame, head_order: Optional[Sequence[str]], head_filter: Optional[str]):
    """Resolve which heads to iterate over for plotting."""
    if head_filter is not None:
        return [head_filter]
    if head_order is not None:
        return list(head_order)
    return [None]


def plot_models_line(
        ax,
        df: pd.DataFrame,
        *,
        x_col: str,
        y_col: str,
        model_order: Sequence[str],
        train_sizes: Iterable[int],
        dataset_markers: dict,
        head_order: Optional[Sequence[str]] = None,
        head_filter: Optional[str] = None,
        y_unc_col: Optional[str] = None,
        line_kwargs: Optional[dict] = None,
        scatter_kwargs: Optional[dict] = None,
        size_scale: float = 1.0,
):
    """Draw model trajectories (line + scatter) on a provided axis.

    Parameters mirror the previous bespoke functions but restrict side
    effects to a single axis. The caller retains control over figure
    creation, layout, legends, and saving.

    Returns
    -------
    list
        The subset of ``model_order`` that were actually plotted.
    """
    active_models = []
    line_cfg = {**DEFAULT_LINE_KWARGS, **(line_kwargs or {})}
    scatter_cfg = {**DEFAULT_SCATTER_KWARGS, **(scatter_kwargs or {})}
    if size_scale != 1.0:
        line_cfg["linewidth"] = line_cfg.get("linewidth", 2) * size_scale
        scatter_cfg["s"] = scatter_cfg.get("s", 140) * size_scale
        scatter_cfg["linewidth"] = scatter_cfg.get("linewidth", 0.7) * size_scale

    train_size_set = set(train_sizes)
    heads_to_plot = _select_heads(df, head_order, head_filter)

    for model in model_order:
        if model not in MODEL_COLORS:
            continue

        df_model = df[df["model"] == model]
        df_model = df_model[df_model["train_size"].isin(train_size_set)]
        if df_model.empty:
            continue

        for head in heads_to_plot:
            if head is not None:
                if "head" not in df_model.columns:
                    continue
                df_head = df_model[df_model["head"] == head]
            else:
                df_head = df_model

            if df_head.empty:
                continue

            df_head = df_head.copy()
            df_head["marker"] = df_head["train_size"].astype(str).map(dataset_markers)
            df_head = df_head.sort_values(x_col)

            xs = df_head[x_col].to_numpy()
            ys = df_head[y_col].to_numpy()
            markers = df_head["marker"].to_numpy()
            y_unc = (
                df_head[y_unc_col].to_numpy()
                if y_unc_col is not None and y_unc_col in df_head
                else np.full_like(ys, np.nan, dtype=float)
            )

            if len(xs) == 0 or np.all(np.isnan(xs)):
                continue

            linestyle = HEAD_LINESTYLES.get(head, "-") if head is not None else "-"
            ax.plot(xs, ys, color=MODEL_COLORS[model], linestyle=linestyle, **line_cfg)

            for x, y, mk in zip(xs, ys, markers):
                ax.scatter(x, y, marker=mk, color=MODEL_COLORS[model], **scatter_cfg)

            if not np.all(np.isnan(y_unc)):
                lower = ys - np.nan_to_num(y_unc, nan=0.0)
                upper = ys + np.nan_to_num(y_unc, nan=0.0)
                ax.fill_between(xs, lower, upper, color=MODEL_COLORS[model], alpha=0.18)

            active_models.append(model)

    deduped = [m for m in model_order if m in set(active_models)]
    return deduped
