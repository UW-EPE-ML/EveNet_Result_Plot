"""
Legend utilities decoupled from plotting code so they can be placed on
any figure or rendered standalone.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from plot_styles.style import MODEL_PRETTY
from plot_styles.core.theme import PlotStyle


def plot_legend(
        fig,
        *,
        active_models: Optional[Sequence[str]] = None,
        train_sizes: Optional[Iterable[int]] = None,
        dataset_markers: Optional[dict] = None,
        dataset_pretty: Optional[dict] = None,
        model_colors: Optional[dict] = None,
        head_order: Optional[Sequence[str]] = None,
        head_linestyles: Optional[dict] = None,
        legends: Optional[List[str]] = None,
        y_start: float = 1.01,
        y_gap: float = 0.07,
        style: PlotStyle | None = None,
        in_figure: bool = False,
):
    """Attach stacked legends above the figure.

    ``legends`` controls which groups are drawn. Valid entries are
    ``"calibration"``, ``"dataset"``, ``"heads"``, ``"models"``.
    """
    if legends is None:
        legends = ["calibration", "dataset", "heads", "models"]

    marker_scale = style.object_scale if style is not None else 1.0
    legend_fontsize = (
        style.legend_size
        if style is not None and style.legend_size is not None
        else plt.rcParams.get("legend.fontsize", 14)
    )

    # Count total legend rows that will be drawn
    n_rows = sum([
        bool("calibration" in legends),
        bool("dataset" in legends and dataset_markers is not None and dataset_pretty is not None),
        bool("heads" in legends and head_order is not None and head_linestyles is not None),
        bool("models" in legends and active_models is not None and model_colors is not None),
    ])

    if in_figure:
        # --- Original behavior: stacked above the axes with fixed spacing ---
        # y_start and y_gap come from function args
        def add_legend(handles, labels, ncol, fontsize=legend_fontsize):
            nonlocal y_start
            leg = fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, y_start),
                ncol=ncol,
                frameon=False,
                fontsize=fontsize,
                handletextpad=0.4,
                columnspacing=1.5,
            )
            # bold section header
            if leg.get_texts():
                leg.get_texts()[0].set_fontweight("bold")
            y_start += y_gap

    else:
        # --- Legend-only canvas: evenly spaced, vertically centered rows ---
        top_gap = 0.02  # space from top of figure
        bottom_gap = 0.02

        usable_height = 1.0 - top_gap - bottom_gap
        y_step = usable_height / max(n_rows, 1)

        row_centers = [
            1.0 - top_gap - (i + 0.5) * y_step
            for i in range(n_rows)
        ]
        row_index = 0

        def add_legend(handles, labels, ncol, fontsize=legend_fontsize):
            nonlocal row_index
            leg = fig.legend(
                handles,
                labels,
                loc="center",
                bbox_to_anchor=(0.5, row_centers[row_index]),
                ncol=ncol,
                frameon=False,
                fontsize=fontsize,
                handletextpad=0.4,
                columnspacing=1.5,
            )
            # bold section header
            if leg.get_texts():
                leg.get_texts()[0].set_fontweight("bold")
            row_index += 1

    if "models" in legends and active_models is not None and model_colors is not None:
        model_handles = [
            Line2D(
                [0],
                [0],
                marker='s',
                markersize=14 * marker_scale,
                linestyle='',
                color=model_colors[m],
                markeredgecolor="black",
            )
            for m in active_models
        ]
        handles = [Line2D([], [], linestyle="none", label="Model Types")] + model_handles
        labels = ["Model Types"] + [MODEL_PRETTY[m] for m in active_models]
        add_legend(handles, labels, ncol=len(model_handles) + 1)

    if "heads" in legends and head_order is not None and head_linestyles is not None:
        head_header = Line2D([], [], linestyle="none", label="Head Types")
        head_handles = [
            Line2D([0], [0], color="black", linestyle=head_linestyles[h], linewidth=2 * marker_scale, label=h)
            for h in head_order
        ]
        handles = [head_header] + head_handles
        labels = ["Head Types"] + list(head_order)
        add_legend(handles, labels, ncol=len(head_handles) + 1)

    if "dataset" in legends and dataset_markers is not None and dataset_pretty is not None:
        ds_handles = [
            Line2D(
                [0],
                [0],
                marker=dataset_markers[str(ds)],
                markersize=14 * marker_scale,
                linestyle='',
                color="gray",
                markeredgecolor="black",
            )
            for ds in train_sizes
        ]
        handles = [Line2D([], [], linestyle="none", label="Dataset Size")] + ds_handles
        labels = ["Dataset Size"] + [dataset_pretty[str(ds)] for ds in train_sizes]
        add_legend(handles, labels, ncol=len(ds_handles) + 1)

    if "calibration" in legends:
        calibrated_handle = Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black")
        uncal_handle = Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch="//")
        handles = [
            Line2D([], [], linestyle="none", label="Calibration"),
            calibrated_handle,
            uncal_handle,
        ]
        labels = ["Calibration", "Calibrated", "Uncalibrated"]
        add_legend(handles, labels, ncol=3)


def plot_only_legend(fig_size=(6, 2), style: PlotStyle | None = None, **kwargs):
    """Convenience wrapper to render legends on an empty canvas."""
    fig = plt.figure(figsize=fig_size)
    plot_legend(fig, style=style, **kwargs)
    return fig
