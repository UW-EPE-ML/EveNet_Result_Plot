"""
Figure-level helper for grid-study loss scatter plots.

This keeps plotting logic reusable while allowing paper_plot to manage
file output and higher-level orchestration.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from plot_styles.core.legend import plot_legend
from plot_styles.core.style_axis import style_axis_basic
from plot_styles.core.theme import PlotStyle, scaled_fig_size, use_style
from plot_styles.style import MODEL_COLORS


def _build_density_contours(
        ax,
        x: np.ndarray,
        y: np.ndarray,
        *,
        bins: int,
        levels: int,
        color: str,
        alpha: float,
        linewidths: float,
        filled: bool,
        use_log_x: bool,
        use_log_y: bool,
):
    if x.size == 0 or y.size == 0:
        return

    x_hist = np.log10(x) if use_log_x else x
    y_hist = np.log10(y) if use_log_y else y

    hist, x_edges, y_edges = np.histogram2d(x_hist, y_hist, bins=bins)
    if not np.any(hist):
        return

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    if use_log_x:
        x_centers = 10 ** x_centers
    if use_log_y:
        y_centers = 10 ** y_centers

    xx, yy = np.meshgrid(x_centers, y_centers)
    zz = hist.T
    zz = np.ma.masked_where(zz <= 0, zz)

    if filled:
        ax.contourf(xx, yy, zz, levels=levels, colors=[color], alpha=alpha)
    else:
        ax.contour(xx, yy, zz, levels=levels, colors=[color], alpha=alpha, linewidths=linewidths)


def plot_grid_loss(
        grid_df,
        *,
        model_specs: Sequence[dict],
        loss_col: str = "min_val_loss",
        raw_step_col: str = "effective_steps_raw",
        per_signal_step_col: str = "effective_steps_per_signal",
        x_label: str = "Effective steps [K]",
        y_label: str = "min val loss",
        xscale: Optional[str] = "log",
        yscale: Optional[str] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        fig_size=(6, 6),
        fig_scale: float = 1.0,
        fig_aspect: Optional[float] = None,
        grid: bool = False,
        individual_style: Optional[dict] = None,
        param_points: Optional[dict] = None,
        param_style: Optional[dict] = None,
        density: Optional[dict] = None,
        legend_config: Optional[dict] = None,
        style: PlotStyle | None = None,
) -> tuple:
    individual_style = individual_style or {}
    param_points = param_points or {}
    param_style = param_style or {}
    density = density or {}
    legend_config = legend_config or {}

    with use_style(style):
        resolved_size = scaled_fig_size(fig_size, scale=fig_scale, aspect_ratio=fig_aspect)
        fig, ax = plt.subplots(figsize=resolved_size)

    active_models = []
    model_colors = {}
    param_legend_entries = []

    for spec in model_specs:
        model_name = spec["model"]
        run_type = spec.get("type")
        label = spec.get("label", model_name)
        color = spec.get("color", MODEL_COLORS.get(f"{model_name}_{run_type}", "black"))
        model_key = spec.get("model_key", f"{model_name}_{run_type}" if run_type else model_name)

        df_model = grid_df[grid_df["model"].eq(model_name)]
        if run_type is not None:
            df_model = df_model[df_model["type"].eq(run_type)]

        if df_model.empty:
            continue

        if run_type == "individual":
            x_vals = df_model[raw_step_col] if raw_step_col in df_model else df_model[per_signal_step_col]
            y_vals = df_model[loss_col]
            mask = np.isfinite(x_vals) & np.isfinite(y_vals)
            x_vals = np.asarray(x_vals[mask], dtype=float)
            y_vals = np.asarray(y_vals[mask], dtype=float)
            if xscale == "log":
                keep = x_vals > 0
                x_vals = x_vals[keep]
                y_vals = y_vals[keep]
            if yscale == "log":
                keep = y_vals > 0
                x_vals = x_vals[keep]
                y_vals = y_vals[keep]

            if density.get("enabled", False):
                _build_density_contours(
                    ax,
                    x_vals,
                    y_vals,
                    bins=density.get("bins", 25),
                    levels=density.get("levels", 6),
                    color=color,
                    alpha=density.get("alpha", 0.25),
                    linewidths=density.get("linewidths", 1.0),
                    filled=density.get("filled", False),
                    use_log_x=density.get("use_log_x", xscale == "log"),
                    use_log_y=density.get("use_log_y", yscale == "log"),
                )

            ax.scatter(
                x_vals,
                y_vals,
                s=individual_style.get("size", 40),
                marker=individual_style.get("marker", "o"),
                color=color,
                alpha=individual_style.get("alpha", 0.6),
                edgecolors=individual_style.get("edgecolor", "none"),
                linewidths=individual_style.get("linewidth", 0.7),
            )
        else:
            loss_values = df_model[loss_col].dropna()
            if loss_values.empty:
                continue
            loss_value = float(loss_values.mean())
            raw_step_vals = df_model[raw_step_col].dropna() if raw_step_col in df_model else df_model[
                per_signal_step_col].dropna()
            per_signal_vals = df_model[per_signal_step_col].dropna()

            raw_step = float(raw_step_vals.mean()) if not raw_step_vals.empty else np.nan
            per_signal_step = float(per_signal_vals.mean()) if not per_signal_vals.empty else np.nan

            for key, x_val in [("raw", raw_step), ("per_signal", per_signal_step)]:
                cfg = param_points.get(key, {})
                if not np.isfinite(x_val) or not np.isfinite(loss_value):
                    continue
                ax.scatter(
                    [x_val],
                    [loss_value],
                    s=cfg.get("size", 220),
                    marker=cfg.get("marker", "*"),
                    color=color if cfg.get("filled", True) else "none",
                    edgecolors=cfg.get("edgecolor", param_style.get("edgecolor", color)),
                    linewidths=param_style.get("linewidth", 1.2),
                    alpha=cfg.get("alpha", param_style.get("alpha", 0.9)),
                    zorder=param_style.get("zorder", 3),
                )

        if model_key not in active_models:
            active_models.append(model_key)
            model_colors[model_key] = color

    if legend_config.get("enabled", True) and active_models:
        param_legend_entries = []
        for _, cfg in (param_points or {}).items():
            param_legend_entries.append(
                {
                    "marker": cfg.get("marker", "*"),
                    "label": cfg.get("label", "Param"),
                    "color": cfg.get("legend_color", "black"),
                    "edgecolor": cfg.get("edgecolor", "black"),
                    "facecolor": cfg.get("legend_facecolor", cfg.get("legend_color", "black")),
                    "alpha": cfg.get("legend_alpha", cfg.get("alpha", 1.0)),
                }
            )

        plot_legend(
            fig,
            active_models=active_models,
            model_colors=model_colors,
            legends=legend_config.get("sections", ["models", "param"]),
            param_entries=param_legend_entries,
            style=style,
            in_figure=True,
            y_start=legend_config.get("y_start", 1.02),
            y_gap=legend_config.get("y_gap", 0.08),
        )

    style_axis_basic(
        ax,
        xscale=xscale,
        yscale=yscale,
        x_label=x_label,
        y_label=y_label,
        y_min=y_min,
        y_max=y_max,
        grid=grid,
        style=style,
    )

    return fig, ax
