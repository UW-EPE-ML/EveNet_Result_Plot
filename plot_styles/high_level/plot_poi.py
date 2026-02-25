"""
POI-style QE summary plot with a broken x-axis:
- left panel: reference threshold at D=1
- right panel: main values + inline table text
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from plot_styles.core.style_axis import apply_nature_axis_style
from plot_styles.core.theme import PlotStyle, scaled_fig_size, use_style
from plot_styles.style import MODEL_COLORS, MODEL_PRETTY


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _coerce_numeric_clean(series: pd.Series) -> pd.Series:
    """Coerce potentially malformed numeric strings (e.g. '0.0.317') to float."""

    def _fix(v):
        if isinstance(v, str):
            s = v.strip().replace(",", "")
            if s.count(".") > 1:
                # Observed malformed values like "0.0.317..."; remove extra first dots.
                while s.count(".") > 1:
                    dot_idx = s.find(".")
                    s = s[:dot_idx] + s[dot_idx + 1 :]
            return s
        return v

    return pd.to_numeric(series.map(_fix), errors="coerce")


def _format_numeric(value: float, precision: int) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{precision}f}"


def _pick_row(df: pd.DataFrame) -> pd.Series:
    """Pick a representative row when duplicates exist for model/train size."""
    if df.empty:
        raise ValueError("Cannot pick a row from an empty DataFrame.")
    if "val_loss" in df.columns:
        return df.sort_values("val_loss", na_position="last").iloc[0]
    return df.iloc[0]


def _draw_axis_break(ax_left, ax_right, *, color: str = "black", size: float = 0.012, linewidth: float = 1.2):
    """Draw diagonal break marks between two horizontally adjacent axes."""
    kwargs = dict(color=color, clip_on=False, linewidth=linewidth)
    ax_left.plot((1 - size, 1 + size), (-size, +size), transform=ax_left.transAxes, **kwargs)
    ax_left.plot((1 - size, 1 + size), (1 - size, 1 + size), transform=ax_left.transAxes, **kwargs)
    ax_right.plot((-size, +size), (-size, +size), transform=ax_right.transAxes, **kwargs)
    ax_right.plot((-size, +size), (1 - size, 1 + size), transform=ax_right.transAxes, **kwargs)


def plot_qe_poi(
    data_df: pd.DataFrame,
    *,
    model_order: Sequence[str],
    train_sizes: Sequence[int],
    dataset_pretty: dict[str, str],
    dataset_markers: dict[str, str] | None = None,
    model_col: str = "model",
    train_size_col: str = "train_size",
    x_col: str = "concurrence",
    xerr_col: str = "uncertainty",
    delta_col: str = "deltaD",
    significance_col: str | None = None,
    d_offset: float = 1.0,
    x_label: str = r"Observable $D$",
    y_label: str = "Model",
    dataset_label_fmt: str = "{pretty}",
    left_panel_xlim: tuple[float, float] = (0.94, 1.04),
    left_indicator_x: float = 1.0,
    left_indicator_text: str = r"Separabel state ($D > 1$)",
    left_indicator_color: str = "#a81228",
    left_indicator_fontsize: float | None = None,
    show_left_shading: bool = True,
    left_shading_color: str = "grey",
    left_shading_alpha: float = 0.22,
    width_ratios: tuple[float, float] = (1.0, 4.0),
    show_row_separators: bool = True,
    right_x_min_factor: float = 0.95,
    right_x_max_factor: float = 1.75,
    text_x_shift_fraction: float = 0.025,
    uncertainty_scale: float = 1.0,
    concurrence_precision: int = 3,
    uncertainty_precision: int = 4,
    delta_precision: int = 2,
    significance_precision: int = 2,  # kept for backward config compatibility
    model_row_gap: float = 1.0,
    intra_group_gap: float = 1.0,
    text_font_family: str | None = None,
    table_text_size: float | None = None,
    col_pad: tuple[int, int, int] = (3, 4, 4),
    header_gap: float = 0.24,
    top_padding: float = 0.10,
    bottom_padding: float = 0.45,
    fig_size: tuple[float, float] = (15, 8),
    fig_scale: float = 1.0,
    fig_aspect: float | None = None,
    style: PlotStyle | None = None,
) -> tuple:
    """Render QE POI summary with the classic broken-axis style."""
    if data_df is None or data_df.empty:
        raise ValueError("`data_df` is empty; cannot render POI plot.")

    required = {model_col, train_size_col, x_col, xerr_col, delta_col}
    missing = required.difference(set(data_df.columns))
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise KeyError(f"QE POI plot is missing required columns: {missing_cols}")

    df = data_df.copy()
    df[x_col] = _coerce_numeric_clean(df[x_col])
    df[xerr_col] = _coerce_numeric_clean(df[xerr_col])
    df[delta_col] = _coerce_numeric_clean(df[delta_col])
    df["_poi_d"] = df[x_col] + float(d_offset)

    if significance_col and significance_col in df.columns:
        df["_poi_sig"] = _coerce_numeric_clean(df[significance_col])
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = df[delta_col].to_numpy(dtype=float)
            df["_poi_sig"] = np.where(delta > 0, 1.0 / delta, np.nan)

    available_sizes = set(pd.to_numeric(df[train_size_col], errors="coerce").dropna().astype(int).tolist())
    resolved_sizes = [int(s) for s in train_sizes if int(s) in available_sizes]
    if not resolved_sizes:
        resolved_sizes = sorted(available_sizes)

    df = df[df[train_size_col].isin(resolved_sizes)]
    active_models = [
        model
        for model in model_order
        if ((df[model_col] == model) & np.isfinite(df["_poi_d"])).any()
    ]
    if not active_models:
        raise ValueError("No finite D values found for requested models/train sizes.")

    with use_style(style):
        resolved_size = scaled_fig_size(fig_size, scale=fig_scale, aspect_ratio=fig_aspect)
        fig, (ax_left, ax_right) = plt.subplots(
            1,
            2,
            figsize=resolved_size,
            sharey=True,
            gridspec_kw={"width_ratios": width_ratios, "wspace": 0.02},
        )

    y_base = np.arange(len(active_models), dtype=float) * float(model_row_gap)
    if len(resolved_sizes) == 1:
        shifts = np.array([0.0], dtype=float)
    else:
        shifts = np.linspace(
            -(len(resolved_sizes) - 1) / (2 * len(resolved_sizes)),
            (len(resolved_sizes) - 1) / (2 * len(resolved_sizes)),
            len(resolved_sizes),
        ) * float(intra_group_gap)

    marker_scale = style.object_scale if style is not None else 1.0
    text_size = (
        style.tick_label_size
        if style is not None and style.tick_label_size is not None
        else plt.rcParams.get("ytick.labelsize", 12)
    )
    if table_text_size is not None:
        text_size = table_text_size
    left_text_size = left_indicator_fontsize if left_indicator_fontsize is not None else max(float(text_size) + 2.0, 14.0)
    row_line_width = 0.8 * marker_scale

    finite_d = []
    finite_unc = []
    for model in active_models:
        for size in resolved_sizes:
            sub = df[(df[train_size_col] == size) & (df[model_col] == model)]
            if sub.empty:
                continue
            row = _pick_row(sub)
            d_val = _safe_float(row.get("_poi_d"))
            u_val = abs(_safe_float(row.get(xerr_col))) * float(uncertainty_scale)
            if np.isfinite(d_val):
                finite_d.append(d_val)
                finite_unc.append(u_val if np.isfinite(u_val) else 0.0)

    if not finite_d:
        raise ValueError("No finite D values available for POI plot.")

    d_arr = np.asarray(finite_d, dtype=float)
    u_arr = np.asarray(finite_unc, dtype=float)
    right_min = max(left_panel_xlim[1] + 0.02, float(np.nanmin(d_arr - u_arr) * right_x_min_factor))
    right_max = float(np.nanmax(d_arr + u_arr) * right_x_max_factor)
    if right_max <= right_min:
        right_max = right_min + 0.5

    max_shift = float(np.max(np.abs(shifts))) if len(shifts) else 0.0
    header_y = (y_base[-1] + max_shift + header_gap) if len(y_base) else header_gap
    y_top = header_y + top_padding
    y_bottom = y_base[0] - bottom_padding if len(y_base) else -bottom_padding

    if show_row_separators and len(y_base) > 1:
        for idx in range(len(y_base) - 1):
            y_sep = 0.5 * (y_base[idx] + y_base[idx + 1])
            ax_left.axhline(y_sep, color="0.75", linestyle="--", linewidth=row_line_width, zorder=0)
            ax_right.axhline(y_sep, color="0.75", linestyle="--", linewidth=row_line_width, zorder=0)

    x_text = right_max - (right_max - right_min) * text_x_shift_fraction
    text_kwargs = {}
    if text_font_family is not None:
        text_kwargs["family"] = text_font_family

    row_entries: list[dict] = []

    for model_idx, model in enumerate(active_models):
        for size_idx, size in enumerate(resolved_sizes):
            marker = (dataset_markers or {}).get(str(size), "o")
            sub = df[(df[train_size_col] == size) & (df[model_col] == model)]
            if sub.empty:
                continue
            row = _pick_row(sub)

            d_val = _safe_float(row.get("_poi_d"))
            unc_val = abs(_safe_float(row.get(xerr_col)))
            unc_plot = unc_val * float(uncertainty_scale)
            delta_val = _safe_float(row.get(delta_col))
            if not np.isfinite(d_val):
                continue

            y_val = y_base[model_idx] + shifts[size_idx]
            color = MODEL_COLORS.get(model, "black")
            axis_main = ax_left if d_val <= left_panel_xlim[1] else ax_right

            if np.isfinite(unc_plot) and unc_plot > 0:
                axis_main.errorbar(
                    d_val,
                    y_val,
                    xerr=np.array([[unc_plot], [unc_plot]]),
                    fmt=marker,
                    markersize=8 * marker_scale,
                    capsize=3.5 * marker_scale,
                    capthick=1.1 * marker_scale,
                    elinewidth=1.5 * marker_scale,
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    zorder=8,
                )
            else:
                axis_main.scatter(
                    [d_val],
                    [y_val],
                    s=64 * marker_scale,
                    marker=marker,
                    color=color,
                    edgecolor=color,
                    zorder=8,
                )

            if np.isfinite(delta_val):
                if delta_val < 10:
                    delta_str = f"{delta_val:.2f}"
                else:
                    delta_str = f"{delta_val:.1f}"
            else:
                delta_str = "n/a"
            size_label = dataset_pretty.get(str(size), f"{size}")
            d_str = f"{d_val:.{concurrence_precision}f}"
            unc_str = f"{unc_plot:.{uncertainty_precision}f}" if np.isfinite(unc_plot) else "n/a"
            row_entries.append(
                {
                    "y": y_val,
                    "color": color,
                    "size_raw": size_label,
                    "size_tex": size_label.replace("%", r"\%"),
                    "d": d_str,
                    "unc": unc_str,
                    "delta": delta_str,
                }
            )

    if row_entries:
        size_header = "Size"
        d_header = "D"
        unc_header = rf"\sigma\times{uncertainty_scale:g}"
        delta_header = rf"D / \sigma [\%]"

        size_w = max(len(size_header), max(len(r["size_raw"]) for r in row_entries))
        d_w = max(len(d_header), max(len(r["d"]) for r in row_entries))
        unc_w = max(len(unc_header), max(len(r["unc"]) for r in row_entries))
        pad_size, pad_d, pad_unc = col_pad

        def _pad_math(text: str, width: int, extra: int) -> str:
            return text + "~" * (max(width - len(text), 0) + extra)

        for r in row_entries:
            row_text = (
                rf"$"
                rf"{_pad_math(r['size_tex'], size_w, pad_size)}"
                rf"{_pad_math(r['d'], d_w, pad_d)}"
                rf"{_pad_math(r['unc'], unc_w, pad_unc)}"
                rf"{r['delta']}"
                rf"$"
            )
            ax_right.text(
                x_text,
                r["y"],
                row_text,
                ha="right",
                va="center",
                fontsize=text_size,
                color=r["color"],
                zorder=9,
                **text_kwargs,
            )

        header_text = (
            rf"$"
            rf"{_pad_math(size_header, size_w, pad_size)}"
            rf"{_pad_math(d_header, d_w, pad_d)}"
            rf"{_pad_math(unc_header, unc_w, pad_unc)}"
            rf"{delta_header}"
            rf"$"
        )
        ax_right.text(
            x_text,
            header_y,
            header_text,
            ha="right",
            va="center",
            fontsize=text_size,
            color="black",
            zorder=10,
            **text_kwargs,
        )

    model_labels = [MODEL_PRETTY.get(model, model) for model in active_models]

    ax_left.set_yticks(y_base)
    ax_left.set_yticklabels(model_labels)
    ax_left.set_ylabel(y_label)

    ax_left.set_xlim(*left_panel_xlim)
    ax_left.set_xticks([left_indicator_x])
    ax_left.set_xticklabels([f"{left_indicator_x:g}"])

    ax_right.set_xlim(right_min, right_max)
    ax_left.set_ylim(y_bottom, y_top)
    ax_right.set_ylim(y_bottom, y_top)

    apply_nature_axis_style(ax_left, style=style)
    apply_nature_axis_style(ax_right, style=style)

    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.tick_params(left=False, labelleft=False)
    _draw_axis_break(ax_left, ax_right, linewidth=1.4 * marker_scale)

    if show_left_shading:
        shade_min = min(left_panel_xlim[0], left_indicator_x)
        shade_max = min(left_panel_xlim[1], left_indicator_x, 1.0)
        if shade_max > shade_min:
            shade_rect = Rectangle(
                (shade_min, y_bottom),
                shade_max - shade_min,
                y_top - y_bottom,
                facecolor=left_shading_color,
                edgecolor="none",
                alpha=left_shading_alpha,
                zorder=1.5,
            )
            ax_left.add_patch(shade_rect)

    ax_left.axvline(
        x=left_indicator_x,
        color=left_indicator_color,
        linestyle="-",
        linewidth=3.0 * marker_scale,
        alpha=0.85,
        zorder=2,
    )
    ax_left.text(
        0.62,
        0.5,
        left_indicator_text,
        ha="left",
        va="center",
        fontsize=left_text_size,
        color=left_indicator_color,
        rotation=90,
        transform=ax_left.transAxes,
    )

    if hasattr(fig, "supxlabel"):
        fig.supxlabel(x_label)
    else:
        fig.text(0.5, 0.01, x_label, ha="center")

    return fig, (ax_left, ax_right), active_models
