from scipy.signal import savgol_filter
from collections import defaultdict
from matplotlib.gridspec import GridSpec
from plot_styles.utils import apply_nature_axis_style, plot_legend, save_axis
import matplotlib.pyplot as plt
import numpy as np
from plot_styles.style import MODEL_COLORS, MODEL_PRETTY, HEAD_LINESTYLES
import os
from matplotlib.transforms import blended_transform_factory
from plot_styles.core.theme import PlotStyle, scaled_fig_size, use_style


def smooth_curve(y, window=31, poly=3):
    """
    Smooth 1D array with Savitzky–Golay filtering.
    Auto-adjust window if the ROC array is short.
    """
    if len(y) < window:
        window = max(5, len(y) // 2 * 2 + 1)
    if window < 5:
        return y
    return savgol_filter(y, window_length=window, polyorder=poly, mode="interp")


def compute_sic_with_unc(TPR, FPR, FPR_unc):
    FPR = np.clip(FPR, 1e-4, 1)
    SIC = TPR / np.sqrt(FPR)
    SIC_unc = TPR * (0.5 / (np.sqrt(FPR) ** 3)) * FPR_unc
    return SIC, SIC_unc


def _collect_sic_data(data_df, model_order, train_sizes, head_order):
    grouped_val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # head → model → max SIC
    grouped_err = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # head → model → max SIC uncertainty
    curve_traces = defaultdict(list)  # head -> list[(TPR, SIC, SIC_unc, model)]
    active_models = []

    for model in model_order:
        if model not in MODEL_COLORS:
            continue

        df_model = data_df[data_df["model"] == model].sort_values("train_size")

        for _, row in df_model.iterrows():
            FPR = np.array(row["FPR"])
            TPR = np.array(row["TPR"])
            FPR_unc = np.array(row["FPR_unc"])

            SIC, SIC_unc = compute_sic_with_unc(TPR, FPR, FPR_unc)

            SIC_smooth = smooth_curve(SIC)
            SIC_unc_smooth = smooth_curve(SIC_unc)

            if row["train_size"] == max(train_sizes) and row["head"] in head_order:
                curve_traces[row["head"]].append((TPR, SIC_smooth, SIC_unc_smooth, model))

            # Compute max SIC on the raw curve
            max_idx = np.argmax(SIC)
            max_sic = SIC[max_idx]
            max_sic_unc = SIC_unc[max_idx]

            head_name = row["head"]
            grouped_val[head_name][model][row["train_size"]] = max_sic
            grouped_err[head_name][model][row["train_size"]] = max_sic_unc

        active_models.append(model)

    ordered_active = [m for m in model_order if m in set(active_models)]
    return curve_traces, grouped_val, grouped_err, ordered_active


def _draw_sic_curve(ax_curve, curve_traces, head_order, style: PlotStyle | None = None):
    for head in head_order:
        for TPR, SIC_smooth, SIC_unc_smooth, model in curve_traces.get(head, []):
            ax_curve.plot(
                TPR,
                SIC_smooth,
                color=MODEL_COLORS[model],
                linestyle=HEAD_LINESTYLES.get(head, "-"),
                linewidth=3 * (style.object_scale if style else 1.0),
            )

            if head == "Cls":
                ax_curve.fill_between(
                    TPR,
                    SIC_smooth - SIC_unc_smooth,
                    SIC_smooth + SIC_unc_smooth,
                    color=MODEL_COLORS[model],
                    alpha=0.2,
                )


def _draw_sic_bar(ax_bar, grouped_val, grouped_err, model_order, head_order, train_sizes,
                  style: PlotStyle | None = None):
    n_heads = len(head_order)
    n_models = len(model_order)
    indices = np.arange(n_heads)
    bar_width = 0.75 / max(1, n_models)

    for i_m, model in enumerate(model_order):
        xb, yb, yerrb = [], [], []

        for i_h, head in enumerate(head_order):
            val = grouped_val.get(head, {}).get(model, {}).get(max(train_sizes), np.nan)
            err = grouped_err.get(head, {}).get(model, {}).get(max(train_sizes), np.nan)

            if np.isnan(val):
                continue

            xb.append(indices[i_h] + (i_m - (n_models - 1) / 2) * bar_width)
            yb.append(val)
            yerrb.append(err)

        if not xb:
            continue

        ax_bar.bar(
            xb,
            yb,
            width=bar_width,
            color=MODEL_COLORS.get(model),
            edgecolor="black",
            alpha=0.9,
            linewidth=0.8 * (style.object_scale if style else 1.0),
            yerr=yerrb,
            capsize=4,
            label=model,
        )

    ax_bar.set_xticks(indices)
    ax_bar.set_xticklabels(head_order)


def _draw_sic_scatter(
        ax_scatter,
        grouped_val,
        grouped_err,
        model_order,
        head_order,
        train_sizes,
        dataset_markers,
        style: PlotStyle | None = None,
        *,
        x_indicator=None,
        x_indicator_text_config=None,
):
    for model in model_order:
        for head in head_order:
            xs, ys, yerr, markers = [], [], [], []
            for train_size in train_sizes:
                val = grouped_val.get(head, {}).get(model, {}).get(train_size, np.nan)
                err = grouped_err.get(head, {}).get(model, {}).get(train_size, np.nan)

                xs.append(train_size)
                ys.append(val)
                yerr.append(err)
                markers.append(dataset_markers.get(str(train_size), "o"))

            ax_scatter.plot(
                xs,
                ys,
                color=MODEL_COLORS.get(model),
                linestyle=HEAD_LINESTYLES.get(head, "-"),
                linewidth=3 * (style.object_scale if style else 1.0),
                alpha=0.9,
            )

            for x, y, mk in zip(xs, ys, markers):
                ax_scatter.scatter(
                    x,
                    y,
                    s=140 * (style.object_scale if style else 1.0),
                    marker=mk,
                    color=MODEL_COLORS.get(model),
                    edgecolor="black",
                    linewidth=0.7 * (style.object_scale if style else 1.0),
                    alpha=0.75,
                )

            if head == "Cls" and any(yerr):
                lower = [y - (u or 0) for y, u in zip(ys, yerr)]
                upper = [y + (u or 0) for y, u in zip(ys, yerr)]
                ax_scatter.fill_between(xs, lower, upper, color=MODEL_COLORS[model], alpha=0.18)

    if x_indicator:
        ax_scatter.axvline(x=x_indicator, color="gray", linestyle="--", linewidth=2, alpha=0.7)
        trans = blended_transform_factory(ax_scatter.transData, ax_scatter.transAxes)
        if x_indicator_text_config:
            ax_scatter.text(
                x_indicator * x_indicator_text_config.get("fraction_x", 0.95),
                x_indicator_text_config.get("fraction_y", 0.025),
                x_indicator_text_config.get("fmt", "Typical HEP dataset: 100k events"),
                fontsize=x_indicator_text_config.get("fontsize", 12),
                color=x_indicator_text_config.get("color", "gray"),
                ha=x_indicator_text_config.get("ha", "right"),
                transform=trans,
            )


def _style_sic_axes(
        ax_curve, ax_bar, ax_scatter, head_order, *, y_min=None, y_max=None,
        bar_style: PlotStyle | None = None,
        scatter_style: PlotStyle | None = None,
        curve_style: PlotStyle | None = None,

):
    ax_curve.set_xlabel("Signal efficiency")
    ax_curve.set_ylabel(r"SIC = $\epsilon_{\rm sig} / \sqrt{\epsilon_{\rm bkg}}$")
    ax_curve.grid(True, linestyle="--", alpha=0.5)
    apply_nature_axis_style(ax_curve, style=curve_style)

    # ax_bar.set_ylabel("Max SIC")
    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.5)
    apply_nature_axis_style(ax_bar, style=bar_style)

    ax_scatter.set_xscale("log")
    ax_scatter.set_xlabel("Train Size [K]")
    ax_scatter.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax_scatter.set_ylabel("Max SIC")
    apply_nature_axis_style(ax_scatter, style=scatter_style)

    if y_min is not None:
        if isinstance(y_min, list) or len(y_min) == 3:
            ax_curve.set_ylim(bottom=y_min[0])
            ax_bar.set_ylim(bottom=y_min[1])
            ax_scatter.set_ylim(bottom=y_min[2])
        else:
            ax_curve.set_ylim(bottom=y_min)
            ax_bar.set_ylim(bottom=y_min)
            ax_scatter.set_ylim(bottom=y_min)
    if y_max is not None:
        if isinstance(y_max, list) or len(y_max) == 3:
            ax_curve.set_ylim(top=y_max[0])
            ax_bar.set_ylim(top=y_max[1])
            ax_scatter.set_ylim(top=y_max[2])
        else:
            ax_curve.set_ylim(top=y_max)
            ax_bar.set_ylim(top=y_max)
            ax_scatter.set_ylim(top=y_max)


def sic_plot(
        data_df,
        model_order,
        train_sizes,
        dataset_markers,
        dataset_pretty,
        head_order,
        y_min=None,
        x_indicator=None,
        fig_size=(20, 6),
        fig_scale: float = 1.0,
        fig_aspect: float | None = None,
        plot_dir=None,
        f_name=None,
        with_legend: bool = True,
        save_individual_axes: bool = False,
        file_format: str = "pdf",
        dpi: int | None = None,
        style: PlotStyle | None = None,
):
    curve_traces, grouped_val, grouped_err, active_models = _collect_sic_data(
        data_df, model_order, train_sizes, head_order
    )

    with use_style(style):
        resolved_size = scaled_fig_size(fig_size, scale=fig_scale, aspect_ratio=fig_aspect)
        fig = plt.figure(figsize=resolved_size)
        gs = GridSpec(1, 3, width_ratios=[1, 1, 1])

        ax_curve = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1])
        ax_scatter = fig.add_subplot(gs[2])

    _draw_sic_curve(ax_curve, curve_traces, head_order, style)
    _draw_sic_bar(ax_bar, grouped_val, grouped_err, model_order, head_order, train_sizes, style)
    _draw_sic_scatter(
        ax_scatter,
        grouped_val,
        grouped_err,
        model_order,
        head_order,
        train_sizes,
        dataset_markers,
        style,
        x_indicator=x_indicator,
    )

    _style_sic_axes(ax_curve, ax_bar, ax_scatter, head_order, y_min=y_min, style=style)

    def _with_ext(name: str) -> str:
        root, ext = os.path.splitext(name)
        return name if ext else f"{root}.{file_format}"

    bitmap_formats = {"png", "jpg", "jpeg", "tiff", "bmp"}
    save_kwargs = {"dpi": dpi} if file_format.lower() in bitmap_formats else {}

    axes = [ax_curve, ax_bar, ax_scatter]

    if save_individual_axes and plot_dir is not None:
        save_axis(
            axes[0],
            plot_dir,
            f_name=_with_ext("sic_curve"),
            dpi=dpi,
        )
        save_axis(
            axes[1],
            plot_dir,
            f_name=_with_ext("sic_bar"),
            dpi=dpi,
        )
        save_axis(
            axes[2],
            plot_dir,
            f_name=_with_ext("sic_scatter"),
            dpi=dpi,
        )

    if with_legend:
        plot_legend(
            fig,
            active_models=active_models,
            train_sizes=train_sizes,
            dataset_markers=dataset_markers,
            dataset_pretty=dataset_pretty,
            model_colors=MODEL_COLORS,
            head_order=head_order,
            head_linestyles=HEAD_LINESTYLES,
            legends=["dataset", "heads", "models"],
            style=style,
        )

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    if f_name is not None:
        os.makedirs(plot_dir, exist_ok=True)
        plot_des = os.path.join(plot_dir, _with_ext(f_name))
        fig.savefig(str(plot_des), bbox_inches="tight", **save_kwargs)
        print(f"Saved figure → {plot_des}")
    return fig, axes, active_models


def sic_plot_individual(
        data_df,
        model_order,
        train_sizes,
        dataset_markers,
        dataset_pretty,
        head_order,
        *,
        y_min=None,
        y_max=None,
        x_indicator=None,
        x_indicator_text_config=None,
        fig_size_curve=(7, 6),
        fig_size_bar=(6, 6),
        fig_size_scatter=(7, 6),
        fig_scale: float = 1.0,
        fig_aspect: float | None = None,
        bar_style: PlotStyle | None = None,
        scatter_style: PlotStyle | None = None,
        curve_style: PlotStyle | None = None,
):
    curve_traces, grouped_val, grouped_err, active_models = _collect_sic_data(
        data_df, model_order, train_sizes, head_order
    )

    with use_style(curve_style):
        curve_size = scaled_fig_size(fig_size_curve, scale=fig_scale, aspect_ratio=fig_aspect)
        fig_curve, ax_curve = plt.subplots(figsize=curve_size)

    with use_style(bar_style):
        bar_size = scaled_fig_size(fig_size_bar, scale=fig_scale, aspect_ratio=fig_aspect)
        fig_bar, ax_bar = plt.subplots(figsize=bar_size)

    with use_style(scatter_style):
        scatter_size = scaled_fig_size(fig_size_scatter, scale=fig_scale, aspect_ratio=fig_aspect)
        fig_scatter, ax_scatter = plt.subplots(figsize=scatter_size)

    _draw_sic_curve(ax_curve, curve_traces, head_order, curve_style)
    _draw_sic_bar(ax_bar, grouped_val, grouped_err, model_order, head_order, train_sizes, bar_style)
    _draw_sic_scatter(
        ax_scatter,
        grouped_val,
        grouped_err,
        model_order,
        head_order,
        train_sizes,
        dataset_markers,
        scatter_style,
        x_indicator=x_indicator,
        x_indicator_text_config=x_indicator_text_config,
    )

    _style_sic_axes(
        ax_curve, ax_bar, ax_scatter, head_order, y_min=y_min, y_max=y_max,
        bar_style=bar_style, curve_style=curve_style, scatter_style=scatter_style,
    )

    return {
        "curve": (fig_curve, ax_curve),
        "bar": (fig_bar, ax_bar),
        "scatter": (fig_scatter, ax_scatter),
    }, active_models
