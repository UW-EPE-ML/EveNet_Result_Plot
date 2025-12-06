"""Clean entry point for generating the final paper figures.

This script centralizes the configuration for QE/BSM/AD plots so you can
quickly tweak models, dataset markers, axis limits, and figure sizing on a
per-task basis without touching the lower-level plotting helpers. Adjust the
`task_configs` or `figure_options` dictionaries below to match the exact
layout and styling needed for your presentation.
"""

from copy import deepcopy
from pathlib import Path

from paper_plot import (
    DEFAULT_AD_CONFIG,
    DEFAULT_BSM_CONFIG,
    DEFAULT_QE_CONFIG,
    PlotStyle,
    plot_final_paper_figures,
    read_ad_data,
    read_bsm_data,
    read_qe_data,
)


ROOT = Path("plot/final_paper")


def build_task_configs():
    """Return mutable copies of the default configs for each task."""

    return {
        "qe": deepcopy(DEFAULT_QE_CONFIG),
        "bsm": deepcopy(DEFAULT_BSM_CONFIG),
        "ad": deepcopy(DEFAULT_AD_CONFIG),
    }


def build_figure_options():
    """Per-figure overrides for sizing, legend visibility, and styles.

    These are intentionally minimal to keep the plotting surface clean. Add or
    modify keys (e.g., `fig_scale`, `fig_aspect`, `with_legend`, `style`) to
    fine-tune each task's figures independently.
    """

    return {
        "qe": {
            "output_root": ROOT,
            "fig_scale": 1.0,
            "fig_aspect": None,
            "with_legend": False,
        },
        "bsm": {
            "output_root": ROOT,
            "fig_scale": 1.0,
            "fig_aspect": None,
            "with_legend": False,
        },
        "ad": {
            "output_root": ROOT,
            "fig_scale": 1.0,
            "fig_aspect": None,
            "with_legend": False,
        },
    }


def main():
    # Load the prepared result tables. Comment out tasks you do not need in a
    # given run to skip their plots entirely.
    qe_data = read_qe_data("data/QE_results_table.csv")
    bsm_data = read_bsm_data("data/BSM")
    ad_data = read_ad_data("data/AD")

    # Baseline styling that can be further overridden per task or per figure.
    base_style = PlotStyle(base_font_size=16, axis_label_size=16, tick_label_size=14, legend_size=13)

    plot_final_paper_figures(
        qe_data=qe_data,
        bsm_data=bsm_data,
        ad_data=ad_data,
        output_root=str(ROOT),
        file_format="pdf",
        include_legends=False,
        style=base_style,
        figure_options=build_figure_options(),
        task_configs=build_task_configs(),
    )


if __name__ == "__main__":
    main()
