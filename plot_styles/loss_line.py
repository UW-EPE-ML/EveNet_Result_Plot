import os
import matplotlib.pyplot as plt
import numpy as np

from plot_styles.utils import apply_nature_axis_style, plot_legend
from plot_styles.style import MODEL_COLORS, HEAD_LINESTYLES


def plot_loss(
        loss_df,
        train_sizes,
        model_order,
        dataset_markers,
        dataset_pretty,
        y_label="Val loss",
        y_min=None,
        x_indicator=None,
        fig_size=(12, 6),
        f_name=None,
        plot_dir="./",
        multi_panel_config=None,
        grid=False,
        with_legend: bool = True,
):
    axes = []
    if multi_panel_config is None:
        fig, ax = plt.subplots(figsize=fig_size)
        axes.append(ax)
    else:
        fig, axes = plt.subplots(
            multi_panel_config['n_rows'], multi_panel_config['n_cols'], figsize=fig_size, sharex=False, sharey=False
        )
        axes = axes.flatten()

    active_models = []
    for panel_idx, ax in enumerate(axes):
        # -------------------------------------------------------
        # Plot each model
        # -------------------------------------------------------
        for model in model_order:

            if model not in MODEL_COLORS:
                continue

            df_model = loss_df[(loss_df["model"] == model)].copy()
            df_model = df_model[df_model["train_size"].isin(train_sizes)]

            if multi_panel_config is not None:
                mp_cfg = multi_panel_config['configs'][panel_idx]
                df_model = df_model[(df_model["head"] == mp_cfg)]

            # Add marker column directly
            df_model["marker"] = df_model["train_size"].astype(str).map(dataset_markers)


            # Sort once by effective_step
            df_model = df_model.sort_values("effective_step")

            # Extract as arrays (no loops!)
            xs = df_model["effective_step"].to_numpy()
            ys = df_model["val_loss"].to_numpy()
            markers = df_model["marker"].to_numpy()

            if len(xs) == 0 or all(np.isnan(xs)):
                continue
            # ---- line ----
            ax.plot(xs, ys,
                    color=MODEL_COLORS[model],
                    linestyle="-" if 'head' not in df_model.columns else HEAD_LINESTYLES.get(df_model['head'].iloc[0], "-"),
                    linewidth=2,
                    alpha=0.9)

            # ---- points ----
            for x, y, mk in zip(xs, ys, markers):
                ax.scatter(
                    x, y,
                    s=140,
                    marker=mk,
                    color=MODEL_COLORS[model],
                    edgecolor="black",
                    linewidth=0.7
                )

            active_models.append(model)

        # -------------------------------------------------------
        # Axes
        # -------------------------------------------------------
        ax.set_xscale("log")
        ax.set_xlabel("Effective steps", fontsize=16)
        ax.set_ylabel(y_label, fontsize=16)
        if y_min is not None:
            ax.set_ylim(bottom=y_min)
        if multi_panel_config is not None:
            mp_cfg = multi_panel_config['configs'][panel_idx]
            ax.set_title(mp_cfg, fontsize=18, weight="bold")

        apply_nature_axis_style(ax)
        ax.tick_params(axis='both', which='major', labelsize=14)

        if grid:
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        # -------------------------------------------------------
        # Indicator Vertical Line
        # -------------------------------------------------------
        if x_indicator:
            ax.axvline(x=x_indicator, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

    active_models = list(set(active_models))
    active_models = [m for m in model_order if m in active_models]

    if with_legend: plot_legend(
        fig, active_models, train_sizes, dataset_markers, dataset_pretty, MODEL_COLORS,
        None if 'head' not in loss_df.columns or multi_panel_config is None else multi_panel_config['configs'],
        HEAD_LINESTYLES,
        legends=["dataset", "heads", "models"]
    )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    if f_name is not None:
        # ---- SAVE HERE ----
        os.makedirs(plot_dir, exist_ok=True)
        plot_des = os.path.join(plot_dir, f_name)
        fig.savefig(plot_des, bbox_inches="tight")
        print(f"Saved figure → {plot_des}")
