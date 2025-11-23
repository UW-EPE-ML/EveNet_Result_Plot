import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from plot_styles.style import MODEL_COLORS, MODEL_PRETTY
from plot_styles.utils import apply_nature_axis_style, plot_legend
from matplotlib.transforms import blended_transform_factory

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
        show_error=True,
        var="median",  # "median" or "mean"
        y_ref=None,
        y_label=r"Median significance $\pm 68\%$ CL",
        y_min=None,
        f_name=None,
        plot_dir="./",
        with_legend: bool = True,
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

    total_bars_per_group = n_models * 2  # cal + uncal
    bar_width = 0.8 / total_bars_per_group
    # -------------------------------------------------------
    # Plotting
    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=fig_size)

    # benchmark_xpos = None
    base_xpos = None
    for i, model in enumerate(model_list):
        for j, calibrated in enumerate([False, True]):

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
    apply_nature_axis_style(ax)
    if y_min is not None:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(bottom=y_min, top=max(y_max := ymax, y_min))

    sns.despine(ax=ax)

    if y_ref is not None:
        x_ref = (base_xpos[1] + bar_width) / (n_channels - 1 + bar_width)

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
            f"Reference Significance: {y_ref:.1f}$\\sigma$",
            fontsize=12,
            color="gray",
            transform=trans,
        )

    # -------------------------------------------------------
    # Legend
    # -------------------------------------------------------
    if with_legend: plot_legend(
        fig,
        active_models=model_list,
        model_colors=MODEL_COLORS,
        legends=["calibration", "models"]
    )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    if f_name is not None:
        # ---- SAVE HERE ----
        os.makedirs(plot_dir, exist_ok=True)
        plot_des = os.path.join(plot_dir, f_name)
        fig.savefig(plot_des, bbox_inches="tight")
        print(f"Saved figure → {plot_des}")


def plot_ad_gen_summary(
        df,
        models_order=None,
        metric_left="mmd",
        metric_right="mean_calibration_difference",
        figsize=(10, 4),
        label_left="MMD",
        label_right=r"$\bar{\Delta}_{\rm calib}$",
        y_min_left=None,
        y_min_right=None,
        f_name=None,
        plot_dir="./",
        with_legend: bool = True,
):
    """
    Two-panel Nature-style bar plot for after-cut metrics.

    - Left panel: metric_left (e.g. "mmd")
    - Right panel: metric_right (e.g. "mean_calibration_difference")
    - X-axis: OS / SS
    - Colors: model types (green/blue scheme)
    - Hatching: calibrated vs uncalibrated
    - Error bars: central 68% interval (16–84% quantiles)
    """

    # -------------------------------------------------------
    # Color scheme: green/blue for models
    # -------------------------------------------------------

    detected = sorted(df["model"].unique())
    if models_order is not None:
        model_list = [m for m in models_order if m in detected]
    else:
        model_list = [m for m in MODEL_COLORS.keys() if m in detected]

    n_methods = len(model_list)

    # -------------------------------------------------------
    # Aggregation: central 68% interval (16–84%) per group
    # -------------------------------------------------------
    def agg(metric):
        result = {}
        for m in model_list:
            for cal in [False, True]:
                group_stats = []
                for g in ["OS", "SS"]:
                    tmp = df[
                        (df["model"] == m) &
                        (df["calibrated"] == cal) &
                        (df["train_type"] == g)
                        ][metric].dropna()

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

    left_data = agg(metric_left)
    right_data = agg(metric_right)

    # -------------------------------------------------------
    # Plotting helpers
    # -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    def plot_panel(ax, metric_data, y_label, y_min=None, percentage=False):
        x_groups = ["OS", "SS"]
        n_bars = n_methods * 2
        bar_width = 0.8 / n_bars

        for i, method in enumerate(model_list):
            for j, cal in enumerate([False, True]):
                centers = [
                    metric_data[(method, cal)][0][0],
                    metric_data[(method, cal)][1][0],
                ]
                err_lows = [
                    metric_data[(method, cal)][0][1],
                    metric_data[(method, cal)][1][1],
                ]
                err_highs = [
                    metric_data[(method, cal)][0][2],
                    metric_data[(method, cal)][1][2],
                ]

                if percentage:
                    centers = [100 * n for n in centers]
                    err_lows = [100 * n for n in err_lows]
                    err_highs = [100 * n for n in err_highs]

                xpos = (
                        np.arange(2)
                        + (i * 2 + j - n_bars / 2 + 0.5) * bar_width
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

        ax.set_xticks(np.arange(2))
        ax.set_xticklabels(x_groups)
        ax.set_ylabel(y_label)
        apply_nature_axis_style(ax)
        if y_min is not None:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(bottom=y_min, top=max(ymax, y_min))
        sns.despine(ax=ax)

    # Left panel
    plot_panel(axes[0], left_data, label_left, y_min=y_min_left)

    # Right panel
    plot_panel(axes[1], right_data, label_right, y_min=y_min_right, percentage=True)

    # -------------------------------------------------------
    # Legend
    # -------------------------------------------------------
    if with_legend: plot_legend(
        fig,
        active_models=model_list,
        model_colors=MODEL_COLORS,
        legends=["calibration", "models"]
    )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    if f_name is not None:
        # ---- SAVE HERE ----
        os.makedirs(plot_dir, exist_ok=True)
        plot_des = os.path.join(plot_dir, f_name)
        fig.savefig(plot_des, bbox_inches="tight")
        print(f"Saved figure → {plot_des}")
