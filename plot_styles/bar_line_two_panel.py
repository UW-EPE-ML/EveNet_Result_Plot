import os
import numpy as np
import matplotlib.pyplot as plt

from plot_styles.utils import apply_nature_axis_style, plot_legend
from plot_styles.style import MODEL_COLORS, MODEL_PRETTY

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
        y_min=0,
        fig_size=(14, 6),
        panel_ratio=(3, 2),  # << tunable left:right panel ratio
        bar_width=0.025,  # << slimmer bars
        bar_spacing=0.035,  # << small gap between bars
        bar_margin=0.015,
        x_range=(6, 6),  # << bars closer to boundary
        x_indicator=None,
        logy=False,
        plot_dir=None,
        f_name=None,
        with_legend: bool = True,
):
    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(1, 2, width_ratios=panel_ratio)

    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1], sharey=ax_left)

    active_models = []
    # -----------------------------------------------------
    # LEFT PANEL — Thin, elegant bars
    # -----------------------------------------------------
    n_models = len(model_order)

    # Place bars near left/right boundaries
    # x0 ∈ [0 + margin] only a single category
    x0 = np.array([bar_margin])

    for i, model in enumerate(model_order):
        val = data_df[(data_df['train_size'] == max(train_sizes)) & (data_df['model'] == model)][metric].values
        if len(val) == 0 or np.isnan(val[0]):
            continue
        val = val[0]

        active_models.append(model)
        # ultra-thin spacing tuned for elegance
        offset = (i - (n_models - 1) / 2) * bar_spacing

        ax_left.bar(
            x0 + offset,
            [val],
            width=bar_width,
            color=MODEL_COLORS[model],
            edgecolor="black",
            linewidth=0.8,
        )

    # no ticks on x-axis
    ax_left.set_xticks([])
    ax_left.set_xlabel("Full dataset size", fontsize=16)
    ax_left.set_ylabel(y_label, fontsize=16)
    ax_left.set_ylim(bottom=y_min)

    apply_nature_axis_style(ax_left)

    # Expand left axis limits so bars nearly touch boundaries
    ax_left.set_xlim(
        x0[0] - bar_width * x_range[0],
        x0[0] + bar_width * x_range[1]
    )

    # -----------------------------------------------------
    # RIGHT PANEL — Curves
    # -----------------------------------------------------
    for model in model_order:
        df_model = data_df[(data_df["model"] == model)].copy()

        # keep only rows with desired train_sizes
        df_model = df_model[df_model["train_size"].isin(train_sizes)]

        # drop NaN metric values
        df_model = df_model.dropna(subset=[metric])

        # add marker column
        df_model["marker"] = df_model["train_size"].astype(str).map(dataset_markers)

        # sort by train_size for plotting
        df_model = df_model.sort_values("train_size")

        # extract arrays
        xs = df_model["train_size"].to_numpy()
        ys = df_model[metric].to_numpy()
        markers = df_model["marker"].to_numpy()

        if len(xs) == 0:
            continue
        # ---- line ----
        ax_right.plot(xs, ys,
                color=MODEL_COLORS[model],
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
                linewidth=0.7
            )
        active_models.append(model)

    # ax_right.set_xscale("log")
    if logy:
        ax_right.set_yscale("log")
        ax_left.set_yscale("log")
    ax_right.set_xlabel("Train Size [K]", fontsize=16)
    ax_right.set_ylim(bottom=y_min)
    # ax_right.set_ylabel(None)

    apply_nature_axis_style(ax_right)

    ax_left.tick_params(axis='both', which='major', labelsize=14)
    ax_right.tick_params(axis='both', which='major', labelsize=14)

    active_models = list(set(active_models))
    active_models = [m for m in model_order if m in active_models]

    if with_legend: plot_legend(fig, active_models, train_sizes, dataset_markers, dataset_pretty, MODEL_COLORS)

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
