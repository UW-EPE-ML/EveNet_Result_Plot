import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from plot_styles.utils import apply_nature_axis_style, plot_legend
from plot_styles.style import MODEL_COLORS, MODEL_PRETTY, HEAD_LINESTYLES

from matplotlib.gridspec import GridSpec
from matplotlib.transforms import blended_transform_factory


def plot_bar_line(
        data_df,
        metric,
        model_order,
        train_sizes,
        dataset_markers,
        dataset_pretty,
        head_order=None,
        y_label="Metric",
        y_min=None,
        fig_size=(14, 6),
        panel_ratio=(3, 2),  # << tunable left:right panel ratio
        bar_width=0.025,  # << slimmer bars
        bar_spacing=0.035,  # << small gap between bars
        bar_margin=0.015,
        x_range=(6, 6),  # << bars closer to boundary
        x_indicator=None,
        logy=False,
        logx=False,
        plot_dir=None,
        f_name=None,
        with_legend: bool = True,
):
    if head_order is None:
        head_order = [None]

    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(1, 2, width_ratios=panel_ratio)

    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1]) #, sharey=ax_left)

    active_models = []

    max_train_size = max(train_sizes)
    n_heads = len(head_order)
    n_models = len(model_order)
    indices = np.arange(n_heads)
    bar_width = 0.75 / max(1, n_models)

    for i_m, model in enumerate(model_order):
        xb, yb, yerrb = [], [], []

        for i_h, head in enumerate(head_order):
            df_model = data_df[(data_df["model"] == model)].copy()
            # keep only rows with desired train_sizes
            df_model = df_model[df_model["train_size"].isin(train_sizes)]
            # select only rows with desired head
            if head is not None:
                df_model = df_model[df_model["head"] == head]
            # drop NaN metric values
            df_model = df_model.dropna(subset=[metric])
            # add marker column
            df_model["marker"] = df_model["train_size"].astype(str).map(dataset_markers)

            # sort by train_size for plotting
            df_model = df_model.sort_values("train_size")

            # extract arrays
            xs = df_model["train_size"].to_numpy()
            ys = df_model[metric].to_numpy()
            yerr = df_model.get(
                f"{metric}_unc",
                pd.Series([np.nan] * len(xs))
            ).to_numpy()
            markers = df_model["marker"].to_numpy()

            if len(xs) == 0 or all(np.isnan(xs)):
                continue
            # ---- line ----
            ax_right.plot(
                xs, ys,
                color=MODEL_COLORS[model],
                linestyle='-' if head is None else HEAD_LINESTYLES[head],
                linewidth=2,
                alpha=0.9)

            # ---- points ----
            for x, y, mk in zip(xs, ys, markers):
                ax_right.scatter(
                    x, y,
                    s=140,
                    marker=mk,
                    color=MODEL_COLORS[model],
                    edgecolor="black",
                    linewidth=0.7,
                    alpha=0.75,
                )

            if any(yerr):
                lower = [y - (u or 0) for y, u in zip(ys, yerr)]
                upper = [y + (u or 0) for y, u in zip(ys, yerr)]
                ax_right.fill_between(xs, lower, upper, color=MODEL_COLORS[model], alpha=0.18)

            if max_train_size in xs:
                xb.append(indices[i_h] + (i_m - (n_models - 1) / 2) * bar_width)
                yb.append(ys[xs.tolist().index(max_train_size)])
                yerrb.append(yerr[xs.tolist().index(max_train_size)])

            active_models.append(model)
        # ---- bars ----
        ax_left.bar(
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

    if logy:
        ax_right.set_yscale("log")
        ax_left.set_yscale("log")
    if logx:
        ax_right.set_xscale("log")
    ax_right.set_xlabel("Train Size [K]", fontsize=16)
    if head_order == [None]:
        ax_left.set_xticks([])
        ax_left.set_xlabel("Full dataset size", fontsize=16)
    else:
        ax_left.set_xticks(indices)
        ax_left.set_xticklabels(head_order)
    ax_left.set_ylabel(y_label, fontsize=16)
    if y_min is not None:
        if isinstance(y_min, tuple) or isinstance(y_min, list):
            ax_right.set_ylim(bottom=y_min[1])
            ax_left.set_ylim(bottom=y_min[0])
        elif isinstance(y_min, float):
            ax_right.set_ylim(bottom=y_min)
            ax_left.set_ylim(bottom=y_min)

    apply_nature_axis_style(ax_left)
    apply_nature_axis_style(ax_right)

    ax_left.tick_params(axis='both', which='major', labelsize=14)
    ax_right.tick_params(axis='both', which='major', labelsize=14)

    active_models = list(set(active_models))
    active_models = [m for m in model_order if m in active_models]

    if with_legend: plot_legend(
        fig, active_models, train_sizes, dataset_markers, dataset_pretty, MODEL_COLORS,
        None if head_order == [None] else head_order, HEAD_LINESTYLES,
        legends=["dataset", "heads", "models"]
    )
    if x_indicator:
        ax_right.axvline(x=x_indicator, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        trans = blended_transform_factory(ax_right.transData, ax_right.transAxes)
        # annotation
        ax_right.text(
            x_indicator * 1.025,
            0.025,
            f"Typical dataset size: {x_indicator:.0f}K",
            fontsize=12,
            color="gray",
            transform=trans,
        )

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    if f_name is not None:
        # ---- SAVE HERE ----
        os.makedirs(plot_dir, exist_ok=True)
        plot_des = os.path.join(plot_dir, f_name)
        fig.savefig(str(plot_des), bbox_inches="tight")
        print(f"Saved figure → {plot_des}")
    return fig, ax_left, ax_right
