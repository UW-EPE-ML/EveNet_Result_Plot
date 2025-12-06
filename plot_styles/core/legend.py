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
):
    """Attach stacked legends above the figure.

    ``legends`` controls which groups are drawn. Valid entries are
    ``"calibration"``, ``"dataset"``, ``"heads"``, ``"models"``.
    """
    if legends is None:
        legends = ["calibration", "dataset", "heads", "models"]

    def add_legend(handles, labels, ncol, fontsize=14):
        nonlocal y_start
        fig.legend(
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
        y_start += y_gap

    if "calibration" in legends:
        calibrated_handle = Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black")
        uncal_handle = Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch="//")
        handles = [
            Line2D([], [], linestyle="none", label="Calibration"),
            calibrated_handle,
            uncal_handle,
        ]
        labels = ["Calibration", "Calibrated", "Uncalibrated"]
        add_legend(handles, labels, ncol=3, fontsize=15)

    if "dataset" in legends and dataset_markers is not None and dataset_pretty is not None:
        ds_handles = [
            Line2D(
                [0],
                [0],
                marker=dataset_markers[str(ds)],
                markersize=14,
                linestyle='',
                color="gray",
                markeredgecolor="black",
            )
            for ds in train_sizes
        ]
        handles = [Line2D([], [], linestyle="none", label="Dataset Size")] + ds_handles
        labels = ["Dataset Size"] + [dataset_pretty[str(ds)] for ds in train_sizes]
        add_legend(handles, labels, ncol=len(ds_handles) + 1, fontsize=15)

    if "heads" in legends and head_order is not None and head_linestyles is not None:
        head_header = Line2D([], [], linestyle="none", label="Head Types")
        head_handles = [
            Line2D([0], [0], color="black", linestyle=head_linestyles[h], linewidth=2, label=h)
            for h in head_order
        ]
        handles = [head_header] + head_handles
        labels = ["Head Types"] + list(head_order)
        add_legend(handles, labels, ncol=len(head_handles) + 1, fontsize=15)

    if "models" in legends and active_models is not None and model_colors is not None:
        model_handles = [
            Line2D(
                [0],
                [0],
                marker='s',
                markersize=14,
                linestyle='',
                color=model_colors[m],
                markeredgecolor="black",
            )
            for m in active_models
        ]
        handles = [Line2D([], [], linestyle="none", label="Model Types")] + model_handles
        labels = ["Model Types"] + [MODEL_PRETTY[m] for m in active_models]
        add_legend(handles, labels, ncol=len(model_handles) + 1, fontsize=15)


def plot_only_legend(fig_size=(6, 2), **kwargs):
    """Convenience wrapper to render legends on an empty canvas."""
    fig = plt.figure(figsize=fig_size)
    plot_legend(fig, **kwargs)
    return fig
