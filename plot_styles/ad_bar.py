import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from plot_styles.style import MODEL_COLORS
from plot_styles.utils import apply_nature_axis_style, plot_legend
from matplotlib.transforms import blended_transform_factory
from plot_styles.core.theme import PlotStyle, scaled_fig_size, use_style

sns.set_theme(style="white", font_scale=1.2)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "pdf.fonttype": 42,
})

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"


def plot_ad_sig_summary(
        df,
        models_order=None,
        channels_order=None,
        fig_size=(10, 5),
        fig_scale: float = 1.0,
        fig_aspect: float | None = None,
        show_error=True,
        var="median",  # "median" or "mean"
        y_ref=None,
        y_label=r"Median significance $\pm 68\%$ CL",
        y_min=None,
        f_name=None,
        plot_dir="./",
        with_legend: bool = True,
        dpi: int | None = None,
        file_format: str | None = None,
        style: PlotStyle | None = None,
        include_uncalibrated: bool = True,
):
    # Which models appear?
    detected = sorted(df["model"].unique())
    if models_order is not None:
        model_list = [m for m in models_order if m in detected]
    else:
        model_list = [m for m in MODEL_COLORS.keys() if m in detected]

    n_models = len(model_list)

    channels = channels_order if channels_order else sorted(df["channel"].unique())
    n_channels = len(channels)

    total_bars_per_group = n_models * 2 if include_uncalibrated else n_models  # cal + uncal
    bar_width = 0.8 / total_bars_per_group
    # -------------------------------------------------------
    # Plotting
    # -------------------------------------------------------
    with use_style(style):
        resolved_size = scaled_fig_size(fig_size, scale=fig_scale, aspect_ratio=fig_aspect)
        fig, ax = plt.subplots(figsize=resolved_size)

    # benchmark_xpos = None
    base_xpos = None
    for i, model in enumerate(model_list):
        cal_list = [False, True] if include_uncalibrated else [True]
        for j, calibrated in enumerate(cal_list):

            vals, lo_err, hi_err = [], [], []

            for ch in channels:
                row = df[
                    (df["model"] == model)
                    & (df["channel"] == ch)
                    & (df["calibrated"] == calibrated)
                    ]

                if row.empty:
                    vals.append(np.nan)
                    lo_err.append(0)
                    hi_err.append(0)
                else:
                    center = row[var].values[0]
                    lower = row["lower"].values[0]
                    upper = row["upper"].values[0]
                    vals.append(center)
                    lo_err.append(center - lower)
                    hi_err.append(upper - center)

            xpos = (
                    np.arange(n_channels)
                    + (i * 2 + j - total_bars_per_group / 2 + 0.5) * bar_width
            ) if include_uncalibrated else (
                    np.arange(n_channels)
                    + (i + j - total_bars_per_group / 2 + 0.5) * bar_width
            )

            if i == 0 and j == 0:
                base_xpos = xpos.copy()

            color = MODEL_COLORS[model]
            hatch = None if calibrated else "//"
            fill_color = color if calibrated else sns.light_palette(color, 3)[1]

            yerr = np.array([lo_err, hi_err])  # shape (2, N)

            ax.bar(
                xpos,
                vals,
                width=bar_width,
                yerr=yerr if show_error else None,
                color=fill_color,
                edgecolor="black",
                hatch=hatch,
                linewidth=0.8,
                error_kw=dict(
                    lw=1.0,
                    capsize=3,
                    capthick=1,
                    ecolor="black",
                ),
            )

    ax.set_xticks(np.arange(n_channels))
    ax.set_xticklabels([
        ch.replace("train-", "").replace("-test-", " → ") for ch in channels
    ], rotation=25, ha="right")
    ax.set_ylabel(y_label)
    apply_nature_axis_style(ax, style=style)
    if y_min is not None:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(bottom=y_min, top=max(y_max := ymax, y_min))

    # sns.despine(ax=ax)

    if y_ref is not None:
        x_ref = (base_xpos[1] + bar_width * 0) / (n_channels - 1 + bar_width)

        ax.axhline(
            y=y_ref,
            xmin=0.0,  # leftmost of axes
            xmax=x_ref,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7
        )

        trans = blended_transform_factory(ax.transAxes, ax.transData)
        # annotation
        ax.text(
            x_ref * 1.05,
            y_ref * 0.98,
            f"Ref. Significance: {y_ref:.1f}$\\sigma$",
            fontsize=19,
            color="gray",
            transform=trans,
        )

    # -------------------------------------------------------
    # Legend
    # -------------------------------------------------------
    if with_legend:
        plot_legend(
            fig,
            active_models=model_list,
            model_colors=MODEL_COLORS,
            legends=["calibration", "models"] if include_uncalibrated else ["models"],
            style=style,
            in_figure=True,
        )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    def _with_ext(name: str) -> str:
        root, ext = os.path.splitext(name)
        if ext:
            return name
        if file_format:
            return f"{root}.{file_format}"
        return f"{root}.pdf"

    if f_name is not None:
        # ---- SAVE HERE ----
        os.makedirs(plot_dir, exist_ok=True)
        plot_des = os.path.join(plot_dir, _with_ext(f_name))
        fig.savefig(plot_des, bbox_inches="tight", dpi=dpi)
        print(f"Saved figure → {plot_des}")


def plot_ad_gen_summary(
        df,
        models_order=None,
        metric="mmd",
        figsize=(10, 4),
        fig_scale: float = 1.0,
        fig_aspect: float | None = None,
        label="MMD",
        train_types: tuple[str, ...] | list[str] = ("OS", "SS"),
        region_gap: float = 0.4,
        include_uncalibrated: bool | None = None,
        y_min=None,
        f_name=None,
        plot_dir="./",
        with_legend: bool = True,
        dpi: int | None = None,
        file_format: str | None = None,
        style: PlotStyle | None = None,
        in_figure: bool = True,
        percentage: bool = False,
):
    """
    Single-panel bar plot for an after-cut metric (e.g., MMD or calibration magnitude).

    - X-axis: configurable training regions (default: OS / SS)
    - Colors: model types (green/blue scheme)
    - Hatching: calibrated vs uncalibrated
    - Error bars: central 68% interval (16–84% quantiles)
    """

    detected = sorted(df["model"].unique())
    if models_order is not None:
        model_list = [m for m in models_order if m in detected]
    else:
        model_list = [m for m in MODEL_COLORS.keys() if m in detected]

    n_methods = len(model_list)

    # user-provided ordering or unique values from data
    if train_types is None:
        x_groups = tuple(sorted(df["train_type"].unique()))
    else:
        x_groups = tuple(train_types)

    # determine which calibration states actually have data to avoid gaps
    available_cal_states = []
    cal_candidates = [False, True]
    if include_uncalibrated is False:
        cal_candidates = [True]

    for cal_state in cal_candidates:
        has_data = any(
            df[
                (df["model"] == m)
                & (df["calibrated"] == cal_state)
                & (df["train_type"].isin(x_groups))
            ][metric].notna().any()
            for m in model_list
        )
        if has_data:
            available_cal_states.append(cal_state)

    if not available_cal_states:
        available_cal_states = [True]

    def agg(metric_name):
        result = {}
        for m in model_list:
            for cal in available_cal_states:
                group_stats = []
                for g in x_groups:
                    tmp = df[
                        (df["model"] == m) &
                        (df["calibrated"] == cal) &
                        (df["train_type"] == g)
                        ][metric_name].dropna()

                    if len(tmp) > 0:
                        q16, q50, q84 = np.nanpercentile(tmp, [16, 50, 84])
                        center = q50
                        err_low = q50 - q16
                        err_high = q84 - q50
                    else:
                        center = np.nan
                        err_low = 0.0
                        err_high = 0.0

                    group_stats.append((center, err_low, err_high))
                result[(m, cal)] = group_stats
        return result

    metric_data = agg(metric)

    resolved_figsize = figsize or (10, 4)
    with use_style(style):
        resolved_size = scaled_fig_size(resolved_figsize, scale=fig_scale, aspect_ratio=fig_aspect)
        fig, ax = plt.subplots(figsize=resolved_size)

    n_cal_states = len(available_cal_states)
    n_bars = max(1, n_methods * n_cal_states)
    bar_width = 0.8 / n_bars

    group_offsets = []
    cumulative_offset = 0.0
    for _ in x_groups:
        group_offsets.append(cumulative_offset)
        cumulative_offset += n_bars * bar_width + region_gap

    for i, method in enumerate(model_list):
        for j, cal in enumerate(available_cal_states):
            centers = [metric_data[(method, cal)][idx][0] for idx in range(len(x_groups))]
            err_lows = [metric_data[(method, cal)][idx][1] for idx in range(len(x_groups))]
            err_highs = [metric_data[(method, cal)][idx][2] for idx in range(len(x_groups))]

            if percentage:
                centers = [100 * n for n in centers]
                err_lows = [100 * n for n in err_lows]
                err_highs = [100 * n for n in err_highs]

            xpos = (
                np.array(group_offsets)
                + (i * n_cal_states + j - n_bars / 2 + 0.5) * bar_width
            )

            color = MODEL_COLORS[method]
            hatch = None if cal else "//"
            fill_color = color if cal else sns.light_palette(color, 3)[1]

            yerr = np.array([err_lows, err_highs])

            ax.bar(
                xpos,
                centers,
                width=bar_width,
                yerr=yerr,
                capsize=3,
                color=fill_color,
                edgecolor="black",
                hatch=hatch,
                linewidth=0.8,
            )

    ax.set_xticks(group_offsets)
    ax.set_xticklabels(x_groups)
    ax.set_ylabel(label)
    apply_nature_axis_style(ax, style=style)
    if y_min is not None:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(bottom=y_min, top=max(ymax, y_min))
    # sns.despine(ax=ax)

    def _with_ext(name: str) -> str:
        root, ext = os.path.splitext(name)
        if ext:
            return name
        if file_format:
            return f"{root}.{file_format}"
        return f"{root}.pdf"

    if with_legend:
        plot_legend(
            fig,
            active_models=model_list,
            model_colors=MODEL_COLORS,
            legends=["calibration", "models"],
            style=style,
            in_figure=in_figure,
        )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    if f_name is not None:
        # ---- SAVE HERE ----
        os.makedirs(plot_dir, exist_ok=True)
        plot_des = os.path.join(plot_dir, _with_ext(f_name))
        fig.savefig(plot_des, bbox_inches="tight", dpi=dpi)
        print(f"Saved figure → {plot_des}")
