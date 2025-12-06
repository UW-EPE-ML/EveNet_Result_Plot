"""
Reusable bar plotting helper used by multi-panel layouts.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, List
import numpy as np
import pandas as pd

from plot_styles.style import MODEL_COLORS

DEFAULT_BAR_KWARGS = {
    "edgecolor": "black",
    "alpha": 0.9,
    "linewidth": 0.8,
    "capsize": 4,
}


def plot_model_bars(
    ax,
    df: pd.DataFrame,
    *,
    metric: str,
    model_order: Sequence[str],
    head_order: Sequence[str],
    train_size_for_bar: int,
    bar_width: Optional[float] = None,
    y_unc_suffix: str = "_unc",
    bar_kwargs: Optional[dict] = None,
) -> List[str]:
    """Plot grouped bars for each model/head at a fixed train size."""
    active_models = []
    n_heads = len(head_order)
    n_models = len(model_order)
    indices = np.arange(n_heads)
    resolved_bar_width = bar_width or 0.75 / max(1, n_models)
    bar_cfg = {**DEFAULT_BAR_KWARGS, **(bar_kwargs or {})}

    for i_m, model in enumerate(model_order):
        df_model = df[(df["model"] == model) & (df["train_size"] == train_size_for_bar)]
        if df_model.empty or model not in MODEL_COLORS:
            continue

        xb, yb, yerrb = [], [], []

        for i_h, head in enumerate(head_order):
            df_head = df_model[df_model["head"] == head] if "head" in df_model.columns else df_model
            df_head = df_head.dropna(subset=[metric])
            if df_head.empty:
                y_val = None
                y_unc = None
            else:
                df_head = df_head.sort_values(metric)
                y_val = df_head[metric].iloc[-1]
                y_unc_col = f"{metric}{y_unc_suffix}"
                y_unc = df_head[y_unc_col].iloc[-1] if y_unc_col in df_head.columns else None

            if y_val is None:
                continue

            xb.append(indices[i_h] + (i_m - (n_models - 1) / 2) * resolved_bar_width)
            yb.append(y_val)
            yerrb.append(y_unc)

        if not xb:
            continue

        yerr_clean = [np.nan if err is None else err for err in yerrb]
        yerr_to_use = None if not yerr_clean or np.all(np.isnan(yerr_clean)) else yerr_clean

        ax.bar(
            xb,
            yb,
            width=resolved_bar_width,
            color=MODEL_COLORS.get(model),
            yerr=yerr_to_use,
            label=model,
            **bar_cfg,
        )
        active_models.append(model)

    ordered_active = [m for m in model_order if m in set(active_models)]
    if head_order == [None]:
        ax.set_xticks([])
    else:
        ax.set_xticks(indices)
        ax.set_xticklabels(head_order)
    return ordered_active
