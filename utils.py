import json
import re
from pathlib import Path

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