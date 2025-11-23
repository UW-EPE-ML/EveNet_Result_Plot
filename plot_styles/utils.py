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


def plot_legend(
        fig, active_models, train_sizes, dataset_markers, dataset_pretty, MODEL_COLORS,
        head_order=None, HEAD_LINESTYLES=None
):
    y_start = 1.01
    y_gap = 0.07

    # -------------------------------------------------------
    # LEGEND 2 — Dataset Sizes
    # -------------------------------------------------------
    ds_handles = [
        plt.Line2D(
            [0], [0],
            marker=dataset_markers[str(ds)],
            markersize=14,
            linestyle='',
            color="gray",
            markeredgecolor="black",
            label=f"{ds}K"
        )
        for ds in train_sizes
    ]

    fig.legend(
        [plt.Line2D([], [], linestyle="none", label=r"$\bf{Dataset\ Size:}$")] + ds_handles,
        [r"$\bf{Dataset\ Size}$"] + [f"{dataset_pretty[str(ds)]}" for ds in train_sizes],
        loc="upper center",
        bbox_to_anchor=(0.5, y_start + 0 * y_gap),
        ncol=len(ds_handles) + 1,
        frameon=False,
        fontsize=15,
        handletextpad=0.4,
        columnspacing=1.5
    )

    # -------------------------------------------------------
    # LEGEND 3 - Heads
    # -------------------------------------------------------
    if head_order is not None and HEAD_LINESTYLES is not None:
        head_header = plt.Line2D(
            [], [], linestyle="none", label=r"$\bf{Head\ Types}$"
        )

        head_handles = [
            plt.Line2D(
                [0], [0],
                color="black",
                linestyle=HEAD_LINESTYLES[h],
                # marker=markers[h],
                # markerfacecolor="black",
                # markeredgecolor="black",
                # markersize=9,
                linewidth=2,
                label=h
            ) for h in head_order
        ]

        fig.legend(
            handles=[head_header] + head_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, y_start + y_gap),
            ncol=len(head_handles) + 1,
            frameon=False,
            fontsize=15,
            handletextpad=0.4,
            columnspacing=1.5
        )

    # -------------------------------------------------------
    # LEGEND 1 — Model Types
    # -------------------------------------------------------
    model_handles = [
        plt.Line2D(
            [0], [0], marker='s',
            markersize=14,
            linestyle='',
            color=MODEL_COLORS[m],
            markeredgecolor="black",
            label=m
        )
        for m in active_models
    ]

    fig.legend(
        [plt.Line2D([], [], linestyle="none", label=r"$\bf{Model\ Types}$")] + model_handles,
        [r"$\bf{Model\ Types}$"] + [m for m in active_models],
        loc="upper center",
        bbox_to_anchor=(0.5, y_start + 2 * y_gap),
        ncol=len(model_handles) + 1,
        frameon=False,
        fontsize=15,
        handletextpad=0.4,
        columnspacing=1.5
    )
