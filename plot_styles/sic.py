from scipy.signal import savgol_filter
from collections import defaultdict
from matplotlib.gridspec import GridSpec
from plot_styles.utils import apply_nature_axis_style, plot_legend, save_axis
import matplotlib.pyplot as plt
import numpy as np
from plot_styles.style import MODEL_COLORS, MODEL_PRETTY, HEAD_LINESTYLES
import os
from matplotlib.transforms import blended_transform_factory


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
        plot_dir=None,
        f_name=None,
        with_legend: bool = True,
        save_individual_axes: bool = False,
        file_format: str = "pdf",
        dpi: int | None = None,
):
    # ============================================================
    # FIGURE with 1:1:1
    # ============================================================
    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1])

    ax_curve = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])
    ax_scatter = fig.add_subplot(gs[2])

    grouped_val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # head → model → max SIC
    grouped_err = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # head → model → max SIC uncertainty

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

            # -------------------------------------------------------
            # SIC Curve Plot (Only for full train_size)
            # -------------------------------------------------------
            if row["train_size"] == max(train_sizes) and row['head'] in head_order:
                ax_curve.plot(
                    TPR,
                    SIC_smooth,
                    color=MODEL_COLORS[model],
                    linestyle=HEAD_LINESTYLES.get(row['head'], '-'),
                    linewidth=2
                )

                if row['head'] == "Cls":
                    ax_curve.fill_between(
                        TPR,
                        SIC_smooth - SIC_unc_smooth,
                        SIC_smooth + SIC_unc_smooth,
                        color=MODEL_COLORS[model],
                        alpha=0.2,
                    )

            # -------------------------------------------------------
            # Bar Plot
            # -------------------------------------------------------
            # Compute max SIC on the raw (unsmoothed) or smooth curve — choose one
            max_idx = np.argmax(SIC)  # or SIC_smooth
            max_sic = SIC[max_idx]
            max_sic_unc = SIC_unc[max_idx]

            # Store into grouped containers by head and model
            head_name = row["head"]
            grouped_val[head_name][model][row["train_size"]] = max_sic
            grouped_err[head_name][model][row["train_size"]] = max_sic_unc

        active_models.append(model)

    # head_order = sorted(grouped_val.keys())
    n_heads = len(head_order)
    n_models = len(model_order)
    indices = np.arange(n_heads)
    bar_width = 0.75 / max(1, n_models)

    for i_m, model in enumerate(model_order):
        xb, yb, yerrb = [], [], []

        for i_h, head in enumerate(head_order):
            xs, ys, yerr, markers = [], [], [], []
            for _, train_size in enumerate(train_sizes):
                val = grouped_val.get(head, {}).get(model, {}).get(train_size, np.nan)
                err = grouped_err.get(head, {}).get(model, {}).get(train_size, np.nan)

                xs.append(train_size)
                ys.append(val)
                yerr.append(err)
                markers.append(dataset_markers.get(str(train_size), "o"))

                if train_size == max(train_sizes):
                    xb.append(indices[i_h] + (i_m - (n_models - 1) / 2) * bar_width)
                    yb.append(val)
                    yerrb.append(err)

            # ---- line ----
            ax_scatter.plot(xs, ys,
                            color=MODEL_COLORS[model],
                            linestyle=HEAD_LINESTYLES[head],
                            linewidth=2,
                            alpha=0.9)

            # ---- points ----
            for x, y, mk in zip(xs, ys, markers):
                ax_scatter.scatter(
                    x, y,
                    s=140,
                    marker=mk,
                    color=MODEL_COLORS[model],
                    edgecolor="black",
                    linewidth=0.7,
                    alpha=0.75,
                )

            # Only Cls gets uncertainty fill
            if head == "Cls" and any(yerr):
                lower = [y - (u or 0) for y, u in zip(ys, yerr)]
                upper = [y + (u or 0) for y, u in zip(ys, yerr)]
                ax_scatter.fill_between(xs, lower, upper, color=MODEL_COLORS[model], alpha=0.18)

        ax_bar.bar(
            xb, yb,
            width=bar_width,
            color=MODEL_COLORS.get(model),
            edgecolor="black",
            alpha=0.9,
            linewidth=0.8,
            yerr=yerrb,
            capsize=4,
            label=model
        )

    # -------------------------------------------------------
    # Axis styles
    # -------------------------------------------------------
    ax_curve.set_xlabel("Signal efficiency")
    ax_curve.set_ylabel(r"SIC = $\epsilon_{\rm sig} / \sqrt{\epsilon_{\rm bkg}}$")
    ax_curve.grid(True, linestyle="--", alpha=0.5)
    apply_nature_axis_style(ax_curve)

    ax_bar.set_xticks(indices)
    ax_bar.set_xticklabels(head_order)
    ax_bar.set_ylabel("Max SIC")
    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.5)
    apply_nature_axis_style(ax_bar)

    ax_scatter.set_xscale("log")
    ax_scatter.set_xlabel("Dataset Size")
    ax_scatter.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    # ax_scatter.set_yscale("log")
    ax_scatter.set_ylabel("Max SIC")
    apply_nature_axis_style(ax_scatter)

    if y_min is not None:
        if isinstance(y_min, list) or len(y_min) == 3:
            ax_curve.set_ylim(bottom=y_min[0])
            ax_bar.set_ylim(bottom=y_min[1])
            ax_scatter.set_ylim(bottom=y_min[2])
        else:
            ax_curve.set_ylim(bottom=y_min)
            ax_bar.set_ylim(bottom=y_min)
            ax_scatter.set_ylim(bottom=y_min)

    if x_indicator:
        ax_scatter.axvline(x=x_indicator, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        trans = blended_transform_factory(ax_scatter.transData, ax_scatter.transAxes)
        # annotation
        ax_scatter.text(
            x_indicator * 0.95,
            0.025,
            f"Typical HEP dataset: {x_indicator:.0f}k events",
            fontsize=12,
            color="gray",
            transform=trans,
            ha="right",
        )

    def _with_ext(name: str) -> str:
        root, ext = os.path.splitext(name)
        return name if ext else f"{root}.{file_format}"

    bitmap_formats = {"png", "jpg", "jpeg", "tiff", "bmp"}
    save_kwargs = {"dpi": dpi} if file_format.lower() in bitmap_formats else {}

    if save_individual_axes:
        save_axis(
            ax_curve,
            plot_dir,
            f_name=_with_ext("sic_curve"),
            dpi=dpi,
        )
        save_axis(
            ax_bar,
            plot_dir,
            f_name=_with_ext("sic_bar"),
            dpi=dpi,
        )
        save_axis(
            ax_scatter,
            plot_dir,
            f_name=_with_ext("sic_scatter"),
            dpi=dpi,
        )

    active_models = list(set(active_models))
    active_models = [m for m in model_order if m in active_models]

    if with_legend: plot_legend(
        fig, active_models, train_sizes, dataset_markers, dataset_pretty, MODEL_COLORS,
        head_order, HEAD_LINESTYLES,
        legends=["dataset", "heads", "models"]
    )

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    if f_name is not None:
        # ---- SAVE HERE ----
        os.makedirs(plot_dir, exist_ok=True)
        plot_des = os.path.join(plot_dir, _with_ext(f_name))
        fig.savefig(str(plot_des), bbox_inches="tight", **save_kwargs)
        print(f"Saved figure → {plot_des}")
    return fig
