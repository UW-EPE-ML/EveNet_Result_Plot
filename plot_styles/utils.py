def apply_nature_axis_style(ax):
    # Reset any seaborn/CMS tick settings
    ax.tick_params(reset=True)

    # =======================
    # Remove all gridlines
    # =======================
    ax.grid(False)

    # =======================
    # Show only left + bottom spines
    # =======================
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color("black")
    ax.spines['bottom'].set_color("black")
    # =======================
    # Ticks: inward, clean
    # =======================
    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        length=5,
        width=1.0,
        labelsize=11,
        pad=3,
        bottom=True,
        top=False,
        left=True,
        right=False
    )

    # =======================
    # Keep labels tight
    # =======================
    ax.xaxis.labelpad = 4
    ax.yaxis.labelpad = 4


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from plot_styles.style import MODEL_PRETTY

def plot_legend(
        fig,
        active_models=None,
        train_sizes=None,
        dataset_markers=None,
        dataset_pretty=None,
        model_colors=None,
        head_order=None,
        head_linestyles=None,
        legends=None,  # <-- ["calibration", "dataset", "heads", "models"]
        y_start=1.01,
        y_gap=0.07,
):
    # Default: show all legends if None provided
    if legends is None:
        legends = ["calibration", "dataset", "heads", "models"]

    # Unified helper: add a legend at current y_start, then shift automatically
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
            columnspacing=1.5
        )
        y_start += y_gap  # shift down for next legend

    # -------------------------------------------------------
    # 1. Calibration Legend
    # -------------------------------------------------------
    if "calibration" in legends:
        calibrated_handle = Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black")
        uncal_handle = Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch="//")

        handles = [
            Line2D([], [], linestyle="none", label=r"$\bf{Calibration}$"),
            calibrated_handle,
            uncal_handle
        ]
        labels = [r"$\bf{Calibration}$", "Calibrated", "Uncalibrated"]
        add_legend(handles, labels, ncol=3, fontsize=15)

    # -------------------------------------------------------
    # 2. Dataset Size Legend
    # -------------------------------------------------------
    if "dataset" in legends and dataset_markers is not None and dataset_pretty is not None:
        ds_handles = [
            Line2D(
                [0], [0],
                marker=dataset_markers[str(ds)],
                markersize=14,
                linestyle='',
                color="gray",
                markeredgecolor="black",
            )
            for ds in train_sizes
        ]

        handles = [Line2D([], [], linestyle="none", label=r"$\bf{Dataset\ Size}$")] + ds_handles
        labels = [r"$\bf{Dataset\ Size}$"] + [dataset_pretty[str(ds)] for ds in train_sizes]

        add_legend(handles, labels, ncol=len(ds_handles) + 1, fontsize=15)

    # -------------------------------------------------------
    # 3. Head Types Legend
    # -------------------------------------------------------
    if "heads" in legends and head_order is not None and head_linestyles is not None:
        head_header = Line2D([], [], linestyle="none", label=r"$\bf{Head\ Types}$")
        head_handles = [
            Line2D(
                [0], [0],
                color="black",
                linestyle=head_linestyles[h],
                linewidth=2,
                label=h
            ) for h in head_order
        ]

        handles = [head_header] + head_handles
        labels = [r"$\bf{Head\ Types}$"] + head_order

        add_legend(handles, labels, ncol=len(head_handles) + 1, fontsize=15)

    # -------------------------------------------------------
    # 4. Model Types Legend
    # -------------------------------------------------------
    if "models" in legends and active_models is not None and model_colors is not None:
        model_handles = [
            Line2D(
                [0], [0],
                marker='s',
                markersize=14,
                linestyle='',
                color=model_colors[m],
                markeredgecolor="black",
            )
            for m in active_models
        ]

        handles = [Line2D([], [], linestyle="none", label=r"$\bf{Model\ Types}$")] + model_handles
        labels = [r"$\bf{Model\ Types}$"] + [MODEL_PRETTY[m] for m in active_models]

        add_legend(handles, labels, ncol=len(model_handles) + 1, fontsize=15)


import os


def save_axis(ax, plot_dir, f_name):
    # Temporarily hide all other axes
    fig = ax.figure
    original_vis = [a.get_visible() for a in fig.axes]
    for a in fig.axes:
        a.set_visible(False)

    ax.set_visible(True)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    # ---- SAVE HERE ----
    os.makedirs(plot_dir, exist_ok=True)
    plot_des = os.path.join(plot_dir, f_name)
    fig.savefig(plot_des, bbox_inches="tight")
    print(f"Saved figure → {plot_des}")

    # Restore visibility
    for a, vis in zip(fig.axes, original_vis):
        a.set_visible(vis)
