"""Minimal demonstration of the modular plotting API."""
import os
import pandas as pd
import matplotlib.pyplot as plt

from plot_styles.core.line_plot import plot_models_line
from plot_styles.core.style_axis import style_axis_basic
from plot_styles.core.legend import plot_legend
from plot_styles.high_level.plot_loss import plot_loss
from plot_styles.high_level.plot_bar_line import plot_bar_line
from plot_styles.style import (
    MODEL_COLORS,
    QE_DATASET_MARKERS,
    QE_DATASET_PRETTY,
)


def build_example_df():
    rows = []
    for model in ["Nominal", "Scratch", "SSL"]:
        for train_size in [15, 148, 1475]:
            rows.append(
                {
                    "model": model,
                    "train_size": train_size,
                    "effective_step": train_size * 10,
                    "val_loss": 1.0 + 0.2 * (train_size / 1475) + 0.05 * (len(model)),
                }
            )
    return pd.DataFrame(rows)


def demo_axis_level(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    active_models = plot_models_line(
        ax,
        df,
        x_col="effective_step",
        y_col="val_loss",
        model_order=["Nominal", "Scratch", "SSL"],
        train_sizes=[15, 148, 1475],
        dataset_markers=QE_DATASET_MARKERS,
    )
    style_axis_basic(ax, xscale="log", y_label="Val loss", x_label="Effective steps")
    plot_legend(
        fig,
        active_models=active_models,
        train_sizes=[15, 148, 1475],
        dataset_markers=QE_DATASET_MARKERS,
        dataset_pretty=QE_DATASET_PRETTY,
        model_colors=MODEL_COLORS,
        legends=["dataset", "models"],
    )
    os.makedirs("examples/output", exist_ok=True)
    fig.savefig("examples/output/demo_axis_level.pdf", bbox_inches="tight")


def demo_layout_wrappers(df):
    fig, axes, active = plot_loss(
        df,
        train_sizes=[15, 148, 1475],
        model_order=["Nominal", "Scratch", "SSL"],
        dataset_markers=QE_DATASET_MARKERS,
        dataset_pretty=QE_DATASET_PRETTY,
        fig_size=(7, 5),
        grid=True,
    )
    plot_legend(
        fig,
        active_models=active,
        train_sizes=[15, 148, 1475],
        dataset_markers=QE_DATASET_MARKERS,
        dataset_pretty=QE_DATASET_PRETTY,
        model_colors=MODEL_COLORS,
        legends=["dataset", "models"],
    )
    os.makedirs("examples/output", exist_ok=True)
    fig.savefig("examples/output/demo_plot_loss.pdf", bbox_inches="tight")

    fig_bl, _, active_bl = plot_bar_line(
        data_df=df,
        metric="val_loss",
        model_order=["Nominal", "Scratch", "SSL"],
        train_sizes=[15, 148, 1475],
        dataset_markers=QE_DATASET_MARKERS,
        dataset_pretty=QE_DATASET_PRETTY,
        y_label="Val loss",
        x_label="Train size [K]",
        logx=True,
    )
    plot_legend(
        fig_bl,
        active_models=active_bl,
        train_sizes=[15, 148, 1475],
        dataset_markers=QE_DATASET_MARKERS,
        dataset_pretty=QE_DATASET_PRETTY,
        model_colors=MODEL_COLORS,
        legends=["dataset", "models"],
    )
    fig_bl.savefig("examples/output/demo_bar_line.pdf", bbox_inches="tight")


def main():
    df = build_example_df()
    demo_axis_level(df)
    demo_layout_wrappers(df)


if __name__ == "__main__":
    main()
