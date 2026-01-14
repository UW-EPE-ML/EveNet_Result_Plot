import math
import re
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from functools import reduce

from plot_styles.high_level.plot_loss import plot_loss
from plot_styles.high_level.plot_grid_loss import plot_grid_loss
from plot_styles.high_level.plot_bar_line import plot_metric_scatter, plot_metric_bar
from plot_styles.high_level.plot_systematics import plot_systematic_scatter
from plot_styles.sic import sic_plot_individual
from plot_styles.ad_bar import plot_ad_sig_summary, plot_ad_gen_summary
from plot_styles.core.legend import plot_legend, plot_only_legend
from plot_styles.core.theme import PlotStyle, scaled_fig_size, use_style
from plot_styles.core.style_axis import apply_nature_axis_style
from plot_styles.grid_unrolled import plot_unrolled_grid_with_winner_and_ratios
from plot_styles.style import MODEL_COLORS, HEAD_LINESTYLES

BITMAP_FORMATS = {"png", "jpg", "jpeg", "tiff", "bmp"}

DEFAULT_LEGEND_STYLE = PlotStyle(legend_size=12.0)
DEFAULT_BAR_STYLE = PlotStyle(tick_label_size=19.0, nbins=2, full_axis=True)
DEFAULT_STYLE = PlotStyle(base_font_size=20.0, tick_label_size=19.0)

DEFAULT_QE_CONFIG = {
    # "train_sizes": [15, 148, 1475, 2950],
    "train_sizes": [15, 148, 1475],
    "models": ["Nominal", "SSL", "Scratch"],
    # "heads": [],
    "legend": {
        "legends": ["dataset", "heads", "models"],
        "fig_size": (5, 0.75),
        "style": DEFAULT_LEGEND_STYLE,
    },
    "loss": {
        "fig_size": (6, 6), "grid": False, "y_min": 0.82,
        "style": DEFAULT_STYLE,
    },
    "pair_scatter": {
        "fig_size": (7.5, 6),
        "metric": "pairing",
        "y_label": "Pairing Efficiency [%]",
        "x_label": "Train Size [K]",
        "y_min": 20.0,
        'y_max': 89,
        "logx": True,
        "x_indicator": 1e3,
        # "x_indicator_text_config": dict(
        #     fraction_x=0.95,
        #     fraction_y=21.5,
        #     fmt="Typical SM dataset: 1M events",
        #     fontsize=19,
        #     color="gray",
        #     ha="right",
        # ),
        "style": DEFAULT_STYLE,
    },
    "pair_bar": {
        "fig_size": (4, 3.5),
        "metric": "pairing",
        # "y_label": "Pairing Efficiency [%]",
        "y_min": 81.0,
        "y_max": 83.0,
        "style": DEFAULT_BAR_STYLE,
    },
    "delta_scatter": {
        "fig_size": (7.5, 6),
        "metric": "deltaD",
        "y_label": r"precision on D [%]",
        "x_label": "Train Size [K]",
        "y_max": 6.1,
        "y_min": 1.0,
        "logx": True,
        "x_indicator": 1e3,
        # "x_indicator_text_config": dict(
        #     fraction_x=0.95,
        #     fraction_y=1.15,
        #     fmt="Typical SM dataset: 1M events",
        #     fontsize=19,
        #     color="gray",
        #     ha="right",
        # ),
        "y_indicator": 5.3,
        "y_indicator_text_config": dict(
            fraction_x=35,
            fraction_y=1.02,
            fmt=r"Reference precision: 5.3%",
            fontsize=19,
            color="gray",
            ha="left",
        ),
        "style": DEFAULT_STYLE,
    },
    "delta_bar": {
        "fig_size": (4, 3.5),
        "metric": "deltaD",
        # "y_label": r"precision on D [%]",
        "y_min": 1.5,
        "y_max": 1.7,
        "style": DEFAULT_BAR_STYLE,
    },
    "systematics": {
        "pairing": {
            "fig_size": (13, 2.25),
            "style": PlotStyle(
                base_font_size=20.0, tick_label_size=19.0, axis_label_size=19.0,
                legend_size=17.0, legend_anchor=(0.88, 0.07), legend_loc="lower right",
            ),
            "x_label": r"$\Delta_\epsilon^\mathrm{pair} = (\epsilon^\mathrm{pair}-\mu_\epsilon^\mathrm{pair})$",
            "cmap": LinearSegmentedColormap.from_list(
                "custom_jes",
                ["#3F5EFB", "#E5ECF6", "#FC466B", ],  # low → mid → high
                N=256
            ),
            "colorbar_label": r"$\alpha_\mathrm{JES}$ [%]",
            "models": ["Nominal", "Scratch"],
            "metric_col": "pairing_norm",
            "unc_col": "pairing_unc",
            # "color_col": "syst_jes",
            "suffix": "pair",
            "label": "Pairing efficiency (normalized)",
            "noise_names": [
                {"name": "jes_only", "label": "JES", "color_col": "syst_jes"},
                {
                    "name": "met_only", "label": "MET", "color_col": "syst_met", "clim_min": 0.0, "clim_max": 5.0,
                    "colorbar_label": r"$E_{T}^\mathrm{miss}$ [GeV]",
                },
            ],
        },
        "precision": {
            "fig_size": (13, 2.25),
            "style": PlotStyle(
                base_font_size=20.0, tick_label_size=19.0, axis_label_size=19.0,
                legend_size=17.0, legend_anchor=(0.88, 0.07), legend_loc="lower right",
            ),
            "x_label": r"$\Delta_D = \sigma_D-\mu_{\sigma_D}$",
            "cmap": LinearSegmentedColormap.from_list(
                "custom_precision",
                ["#3F5EFB", "#E5ECF6", "#FC466B", ],
                N=256
            ),
            "colorbar_label": r"$\alpha_\mathrm{JES}$ [%]",
            "models": ["Nominal", "Scratch"],
            "metric_col": "precision_norm",
            "unc_col": "precision_unc",
            # "color_col": "syst_jes",
            "suffix": "precision",
            "label": "Precision (normalized)",
            "noise_names": [
                {"name": "jes_only", "label": "JES", "color_col": "syst_jes"},
                {
                    "name": "met_only", "label": "MET", "color_col": "syst_met", "clim_min": 0.0, "clim_max": 5.0,
                    "colorbar_label": r"$E_{T}^\mathrm{miss}$ [GeV]",
                },
            ],
        },
    },
}

DEFAULT_BSM_CONFIG = {
    "train_sizes": [10, 30, 100, 300],
    "typical_dataset_size": 300,
    "models": ["Nominal", "SSL", "Scratch", "SPANet"],
    "heads": ["Cls", "Cls+Asn"],
    "pair_heads": ["Cls+Asn"],
    "legend": {
        "legends": ["dataset", "heads", "models"],
        "fig_size": (6.5, 1.2),
        "style": DEFAULT_LEGEND_STYLE,
    },
    "loss": {
        "fig_size": (6, 6), "grid": False,
        "style": PlotStyle(base_font_size=20.0, tick_label_size=19.0)
    },
    "pair_scatter": {
        "fig_size": (7.5, 6),
        "metric": "pairing",
        "y_label": "Pairing Efficiency [%]",
        "x_label": "Train Size [K]",
        "y_min": 0.0,
        "logx": True,
        "x_indicator": 200,
        # "x_indicator_text_config": dict(
        #     fraction_x=0.95,
        #     fraction_y=20.75,
        #     fmt="Typical BSM dataset: 100K events",
        #     fontsize=19,
        #     color="gray",
        #     ha="right",
        # ),
        "style": DEFAULT_STYLE,
    },
    "pair_bar": {
        "fig_size": (4, 3.5),
        "metric": "pairing",
        # "y_label": "Pairing Efficiency [%]",
        "y_min": 65.0,
        "y_max": 81.0,
        "style": DEFAULT_BAR_STYLE,
    },
    "sic": {
        "x_indicator": 200,
        # "x_indicator_text_config": dict(
        #     fraction_x=0.95,
        #     fraction_y=0.020,
        #     fmt="Typical BSM dataset: 100K events",
        #     fontsize=19,
        #     color="gray",
        #     ha="right",
        # ),
        "y_min": [0, 0, 0.75],
        "y_max": [5.5, 5.5, 6.5],
        "style": DEFAULT_STYLE,
        "fig_size_bar": (4, 3.5),
        "fig_size_scatter": (7.5, 6),
        "fig_size_curve": (7.5, 6),
        "bar_style": DEFAULT_BAR_STYLE,
        "scatter_style": DEFAULT_STYLE,
        "curve_style": DEFAULT_STYLE,
    },
    "systematics": {
        "sic": {
            "fig_size": (13, 2.25),
            "style": PlotStyle(
                base_font_size=20.0, tick_label_size=19.0, axis_label_size=19.0,
                legend_size=17.0, legend_anchor=(0.13, 0.07), legend_loc="lower left",
            ),
            "x_label": r"$\Delta_\mathrm{SIC} = (\mathrm{SIC}-\mu_\mathrm{SIC})/\sigma_\mathrm{SIC}$",
            # "cmap": "coolwarm",
            "cmap": LinearSegmentedColormap.from_list(
                "custom_jes",
                ["#3F5EFB", "#E5ECF6", "#FC466B", ],  # low → mid → high
                N=256
            ),
            "colorbar_label": r"$\alpha_\mathrm{JES}$ [%]",
            "models": ["Nominal", "Scratch"],
            "metric_col": "sic_norm",
            "unc_col": "sic_unc",
            "color_col": "jes_shift_percent",
            "suffix": "sic",
            "label": "SIC (normalized)",
        },
        "pairing": {
            "fig_size": (13, 2.25),
            "style": PlotStyle(
                base_font_size=20.0, tick_label_size=19.0, axis_label_size=19.0,
                legend_size=17.0, legend_anchor=(0.88, 0.07), legend_loc="lower right",
            ),
            "x_label": r"$\Delta_\epsilon^\mathrm{pair} = (\epsilon^\mathrm{pair}-\mu_\epsilon^\mathrm{pair})$",
            # "cmap": "coolwarm",
            "cmap": LinearSegmentedColormap.from_list(
                "custom_jes",
                ["#3F5EFB", "#E5ECF6", "#FC466B", ],  # low → mid → high
                N=256
            ),
            "colorbar_label": r"$\alpha_\mathrm{JES}$ [%]",
            "models": ["Nominal", "Scratch"],
            "metric_col": "pairing_norm",
            "unc_col": "pairing_eff_unc_percent",
            "color_col": "jes_shift_percent",
            "suffix": "pair",
            "label": "Pairing efficiency (normalized)",
        },
    },
}

DEFAULT_AD_CONFIG = {
    "models": ["Nominal", "SSL", "Scratch"],
    "heads": [],
    "legend": {
        "legends": ["calibration", "models"],
        "fig_size": (5, 0.5),
        "style": DEFAULT_LEGEND_STYLE,
    },
    "sig": {
        "channels_order": [
            "train-OS-test-OS",
            "train-SS-test-OS",
            "train-OS-test-SS",
            "train-SS-test-SS",
        ],
        "show_error": True,
        "var": "median",
        "y_ref": 6.4,
        "f_name": "ad_significance",
        "style": PlotStyle(
            base_font_size=18.0, tick_label_size=17.0, legend_size=16.0,
            cms_label_fontsize=20.0, cms_label_y_start=0.98,
            full_axis=True
        ),
        "fig_size": (13, 6),
        "with_legend": False,
        "include_uncalibrated": False,
    },
    "gen_mmd": {
        "metric": "mmd",
        "label": "MMD",
        "f_name": "ad_generation_mmd",
        "fig_size": (6, 4),
        "y_min": 0.6,
        "train_types": ["OS", "SS"],
        "region_gap": 0.4,
    },
    "gen_calibration": {
        "metric": "mean_calibration_difference",
        "label": "Cal. Mag. [%]",
        "f_name": "ad_generation_calibration",
        "fig_size": (6, 4),
        "y_min": 0,
        "percentage": True,
        "train_types": ["OS"],
        "region_gap": 0.4,
        "include_uncalibrated": False,
        "with_legend": False,
        "style": PlotStyle(base_font_size=20.0, tick_label_size=19.0, full_axis=True),
    },
}

DEFAULT_GRID_CONFIG = {
    "metric_col": "max_sic",
    "unc_col": "max_sic_unc",
    "output_subdir": "Grid",
    "base_plot_config": {
        "models": [],
        "figsize": (15, 7),
        "hspace": 0.05,
        "height_ratios": {
            "main": 2.0,
            "aux": 1.2,
            "cutflow": 0.8,
            "winner": 0.5,
            "ratio": 1.0,
        },
        "x_top_label": r"$m_Y$ [GeV]",
        "x_bottom_label": r"$m_X$ [GeV]",
        "y_main_label": "Max SIC",
        "y_main_log": True,
        "tick_fontsize_top": 13,
        "tick_fontsize_bottom": 11,
        "tick_rotation_top": 90,
        "label_fontsize": None,
        "label_fontsize_top": 16,
        "label_fontsize_bottom": 16,
        "label_fontsize_y": 14,
        "x_top": {
            "show": True,
            "show_text": True,
            "show_ticks": True,
            "label_every": 2,
        },
        "block_separator": {
            "enabled": True,
            "color": "0.6",
            "linestyle": "--",
            "linewidth": 1.5,
        },
        "line": {
            "marker": "o",
            "markersize": 4,
            "linewidth": 2.5,
        },
        "style": None,
        "apply_axis_style": True,
        "legend_main": {
            "enabled": False,
            "ncols": 4,
            "fontsize": None,
            "loc": "lower right",
        },
        "winner": {
            "enabled": True,
            "ylabel": "",
            "ylabel_rotation": 90,
            # "ylabel_pad": 37,
        },
        "cutflow": {
            "enabled": True,
            "ylabel": "Signal \n Stats. [K]",
            "color": "0.7",
            "edgecolor": "0.2",
            "alpha": 0.75,
            "linewidth": 1.0,
            "bar_width": 1.0,
            "zorder": 1,
            "log": False,
        },
        "aux_panel": {
            "enabled": True,
            "metric_col": "effective_steps",
            "ylabel": "Updates [K]",
            "y_log": False,
        },
        "ratios": [],
        "ratio_line": {
            "marker": "o",
            "markersize": 4,
            "linewidth": 2.5,
        },
        "unc": {
            "enabled": True,
            "models": [],
            "alpha": 0.45,
            "zorder": 1,
        },
        "boost": {
            "enabled": False,
            "B0": 1.5,
            "B1": 3.0,
            "mH": 125.0,
            "base_color": "gray",
            "alpha_min": 0.0,
            "alpha_max": 0.2,
            "zorder": 0.5,
        },
        # "tight_layout": {"pad": 0.2},
        "subplot_adjust": {
            "top": 0.83, "bottom": 0.08,
            "left": 0.1, "right": 0.98,
        },
    },
    "plots": {
        "individual": {
            "output_name": "grid_sic_individual",
            "title": "Grid SIC (individual)",
            "series": [
                {
                    "model": "evenet-pretrain",
                    "type": "individual",
                    "label": "Full",
                    "color": MODEL_COLORS["evenet-pretrain_individual"],
                },
                {
                    "model": "evenet-scratch",
                    "type": "individual",
                    "label": "Scratch",
                    "color": MODEL_COLORS["evenet-scratch_individual"],
                },
                {
                    "model": "xgb",
                    "type": "individual",
                    "label": "XGBoost",
                    "color": MODEL_COLORS["xgb_individual"],
                },
                {
                    "model": "tabpfn",
                    "type": "individual",
                    "label": "TabPFN v2.5",
                    "color": MODEL_COLORS["tabpfn_individual"],
                },
            ],
            "plot_config": {
                "figsize": (15, 8),
                "height_ratios": {
                    "main": 2.2,
                    "aux": 1.0,
                    "cutflow": 0.8,
                    "winner": 0.5,
                    "ratio": 1.0,
                },
                "subplot_adjust": {
                    "top": 0.865, "bottom": 0.08,
                    "left": 0.1, "right": 0.98,
                },
                "aux_panel": {
                    "enabled": True,
                    "ylabel": "Effective \n Steps [K]",
                },
                "apply_axis_style": True,
                "ratios": [
                    # {"baseline": "XGBoost", "mode": "ratio", "ylabel": "/XGB", "reference_line": True},
                    # {"baseline": "TabPFN v2.5", "mode": "ratio", "ylabel": "/TabPFN", "reference_line": True},
                    {"baseline": "Full", "mode": "ratio", "ylabel": "Ratio \n to Full", "reference_line": True},
                ],
                "unc": {"enabled": True},
            },
        },
        "mixed": {
            "output_name": "grid_sic_mixed",
            "title": "Grid SIC (pretrain individual + parametrized)",
            "series": [
                {
                    "model": "evenet-pretrain",
                    "type": "individual",
                    "label": "Full",
                    "color": MODEL_COLORS["evenet-pretrain_individual"],
                },
                {
                    "model": "evenet-pretrain",
                    "type": "param",
                    "label": "Full (param)",
                    "color": MODEL_COLORS["evenet-pretrain_param"],
                },
                {
                    "model": "evenet-scratch",
                    "type": "param",
                    "label": "Scratch (param)",
                    "color": MODEL_COLORS["evenet-scratch_param"],
                },
                {
                    "model": "xgb",
                    "type": "param",
                    "label": "XGBoost (param)",
                    "color": MODEL_COLORS["xgb_param"],
                },
            ],
            "plot_config": {
                "figsize": (15, 6),
                "hspace": 0.05,
                "ratios": [
                    # {"baseline": "XGBoost (param)", "mode": "ratio", "ylabel": "/XGB", "reference_line": True, "y_log": True},
                    # {"baseline": "EveNet-Scratch (param)", "mode": "ratio", "ylabel": "/Scratch", "reference_line": True, "y_log": True},
                    {"baseline": "Full", "mode": "ratio", "ylabel": "Ratio \n to Full", "reference_line": True,
                     "y_log": False},
                ],
                "unc": {"enabled": True},
                "aux_panel": {
                    "enabled": False,
                },
                "cutflow": {
                    "enabled": False,
                },
                "subplot_adjust": {
                    "top": 0.81, "bottom": 0.10,
                    "left": 0.09, "right": 0.98,
                },
            },
        },
    },
    "loss": {
        "enabled": True,
        "output_name": "grid_loss",
        "title": "Grid loss (EveNet)",
        "loss_col": "min_val_loss",
        "raw_step_col": "effective_steps_raw",
        "per_signal_step_col": "effective_steps_per_signal",
        "x_label": "Effective steps [K]",
        "y_label": "min val loss",
        "xscale": "log",
        "fig_size": (6, 6),
        "grid": False,
        "models": [
            {
                "model": "evenet-pretrain",
                "type": "individual",
                "label": "Full (individual)",
                "color": MODEL_COLORS["evenet-pretrain_individual"],
            },
            {
                "model": "evenet-pretrain",
                "type": "param",
                "label": "Full (param)",
                "color": MODEL_COLORS["evenet-pretrain_param"],
            },
            {
                "model": "evenet-scratch",
                "type": "individual",
                "label": "Scratch (individual)",
                "color": MODEL_COLORS["evenet-scratch_individual"],
            },
            {
                "model": "evenet-scratch",
                "type": "param",
                "label": "Scratch (param)",
                "color": MODEL_COLORS["evenet-scratch_param"],
            },
        ],
        "individual": {
            "marker": "o",
            "size": 40,
            "alpha": 0.55,
            "edgecolor": "none",
        },
        "param_points": {
            "raw": {
                "marker": "*",
                "size": 220,
                "label": "Param (total steps)",
                "filled": True,
                "alpha": 0.9,
                "legend_alpha": 0.9,
            },
            "per_signal": {
                "marker": "X",
                "size": 200,
                "label": "Param / #signals",
                "filled": True,
                "alpha": 0.55,
                "legend_alpha": 0.55,
            },
        },
        "param": {
            "edgecolor": "black",
            "linewidth": 0.9,
            "alpha": 0.9,
            "zorder": 3,
        },
        "density": {
            "enabled": False,
            "bins": 24,
            "levels": 6,
            "alpha": 0.25,
            "linewidths": 1.0,
            "filled": False,
            "use_log_x": True,
            "use_log_y": False,
        },
        "legend": {
            "enabled": True,
            "sections": ["models", "param"],
            "y_start": 1.02,
            "y_gap": 0.08,
        },
        "style": DEFAULT_STYLE,
    },
}


def _with_ext(name: str, file_format: str) -> str:
    root, ext = os.path.splitext(name)
    return name if ext else f"{root}.{file_format}"


def _save_kwargs(file_format: str, dpi: int | None):
    if file_format.lower() in BITMAP_FORMATS and dpi is not None:
        return {"dpi": dpi}
    return {}


def _resolve_style(base: PlotStyle | None, override: PlotStyle | dict | None) -> PlotStyle | None:
    """Return a figure style with optional overrides.

    Parameters
    ----------
    base : PlotStyle | None
        Global default style used when no overrides are provided.
    override : PlotStyle | dict | None
        Either an explicit :class:`PlotStyle` or a mapping of constructor
        keyword arguments (e.g., ``{"base_font_size": 16, "object_scale": 1.2}``).
    """

    if override is None:
        return base
    if isinstance(override, PlotStyle):
        return override

    base_style = base or PlotStyle()
    return PlotStyle(
        base_font_size=override.get("base_font_size", base_style.base_font_size),
        axis_label_size=override.get("axis_label_size", base_style.axis_label_size),
        tick_label_size=override.get("tick_label_size", base_style.tick_label_size),
        legend_size=override.get("legend_size", base_style.legend_size),
        title_size=override.get("title_size", base_style.title_size),
        figure_scale=override.get("figure_scale", base_style.figure_scale),
        object_scale=override.get("object_scale", base_style.object_scale),
    )


def _merge_configs(default: dict, override: dict | None) -> dict:
    """Deep-merge a user override into a default config without mutation."""

    if override is None:
        return default

    merged = {}
    for key, value in default.items():
        if key in override:
            if isinstance(value, dict) and isinstance(override[key], dict):
                merged[key] = _merge_configs(value, override[key])
            else:
                merged[key] = override[key]
        else:
            merged[key] = value

    for key, value in override.items():
        if key not in merged:
            merged[key] = value

    return merged


def plot_task_legend(
        *,
        plot_dir: str,
        model_order,
        train_sizes,
        dataset_markers,
        dataset_pretty,
        head_order=None,
        legends=None,
        file_format: str = "pdf",
        dpi: int | None = None,
        f_name: str = "legend",
        fig_size: tuple[float, float] = (7.5, 2.5),
        fig_scale: float | None = None,
        fig_aspect: float | None = None,
        style: PlotStyle | None = None,
):
    """Render a standalone legend shared by a task's plots."""

    scale = fig_scale if fig_scale is not None else (style.figure_scale if style else 1.0)
    resolved_size = scaled_fig_size(fig_size, scale=scale, aspect_ratio=fig_aspect)
    with use_style(style):
        fig = plot_only_legend(
            fig_size=resolved_size,
            active_models=model_order,
            train_sizes=train_sizes,
            dataset_markers=dataset_markers,
            dataset_pretty=dataset_pretty,
            model_colors=MODEL_COLORS,
            head_order=head_order,
            head_linestyles=HEAD_LINESTYLES,
            legends=legends,
            style=style,
        )

    os.makedirs(plot_dir, exist_ok=True)
    legend_path = os.path.join(plot_dir, _with_ext(f_name, file_format))
    fig.savefig(legend_path, **_save_kwargs(file_format, dpi))
    return legend_path


def convert_epochs_to_steps(epoch, train_size, batch_size_per_GPU, GPUs):
    effective_step = (epoch * train_size * 1000 / (batch_size_per_GPU * GPUs))
    return effective_step


def _iter_systematic_noise_cases(metric_name: str, syst_cfg: dict, data=None):
    """Yield systematic plotting cases including optional noise_name filters.

    Each case carries the suffix used for saving plots, the color column, and
    the filtered dataset (if a ``noise_name`` is specified and available).
    """

    base_suffix = syst_cfg.get("suffix", metric_name)
    noise_entries = syst_cfg.get("noise_names") or [None]

    for entry in noise_entries:
        extra = {}

        if isinstance(entry, dict):
            extra = dict(entry)  # shallow copy, safe
            noise_name = extra.get("name")
            noise_label = extra.get("label", noise_name)
            color_col = extra.get(
                "color_col",
                syst_cfg.get("color_col", "jes_shift_percent")
            )
            suffix = extra.get("suffix") or (
                f"{base_suffix}_{noise_name}" if noise_name else base_suffix
            )
        else:
            noise_name = entry
            noise_label = entry
            color_col = syst_cfg.get("color_col", "jes_shift_percent")
            suffix = f"{base_suffix}_{noise_name}" if noise_name else base_suffix

        subset = data
        if (
                data is not None
                and noise_name is not None
                and isinstance(data, pd.DataFrame)
                and "noise_name" in data.columns
        ):
            subset = data[data["noise_name"] == noise_name]

        yield {
            '_EXTRA': {**extra},  # 👈 pass through arbitrary user-defined keys
            "suffix": suffix,
            "color_col": color_col,
            "noise_name": noise_name,
            "noise_label": noise_label,
            "label": syst_cfg.get("label", metric_name),
            "data": subset,
        }


def _prepare_systematic_cases(systematics_cfg: dict, systematics_data):
    """Expand systematic config into concrete plotting cases."""

    cases = []
    for metric_name, syst_cfg in systematics_cfg.items():
        for case in _iter_systematic_noise_cases(metric_name, syst_cfg, data=systematics_data):
            if case.get("data") is None:
                continue
            if hasattr(case.get("data"), "empty") and case["data"].empty:
                continue
            case.update({"metric_name": metric_name, "config": syst_cfg})
            cases.append(case)
    return cases


def read_qe_data(file_path):
    with open(file_path, 'r') as f:
        data = pd.read_csv(f)

    # Convert epochs to effective steps
    data['effective_step'] = data.apply(
        lambda row: convert_epochs_to_steps(
            row['epoch'],
            row['train_size'],
            batch_size_per_GPU=1024,
            GPUs=16
        ),
        axis=1
    )

    ### systematics
    def collect_results(base_dir: Path, model_dirs: list[Path]):
        rows = []

        for base in [base_dir / d for d in model_dirs]:
            base = Path(base)
            if not base.exists():
                continue

            for noise_dir in base.iterdir():
                if not noise_dir.is_dir():
                    continue

                model_name_map = {
                    "scratch": "Scratch",
                    "full": "Nominal",
                }

                model_name = model_name_map[base.name]
                noise_name = noise_dir.name

                for syst_dir in noise_dir.glob("merged_syst_*"):
                    try:
                        # ---------- read files ----------
                        with open(syst_dir / "event_counts.json") as f:
                            event_counts = json.load(f)
                        with open(syst_dir / "systematics.json") as f:
                            syst = json.load(f)
                        unfolding = pd.read_csv(syst_dir / "unfolding.csv")
                        # ---------- pairing ----------
                        k = event_counts["All Four True"]["Count"]
                        N = event_counts["All Four True"]["Total"]

                        p = k / N
                        pairing = 100.0 * p
                        pairing_unc = 100.0 * math.sqrt(p * (1.0 - p) / N)
                        # ---------- concurrence + uncertainty ----------
                        conc_row = unfolding.loc[
                            unfolding["name"] == "Concurrence"
                            ].iloc[0]
                        unc_up = conc_row["uncertainty_up"]
                        unc_dn = conc_row["uncertainty_down"]
                        concurrence = conc_row["value"]
                        precision = (unc_up + unc_dn) / 2.0 / concurrence
                        rows.append({
                            "model_name": model_name,
                            "model_pretty": model_name,
                            "noise_name": noise_name,
                            "syst_jes": syst.get("syst_jes", 0.0) * 100,
                            "syst_met_px": syst.get("syst_met_px", 0.0),
                            "syst_met_py": syst.get("syst_met_py", 0.0),
                            "syst_met": math.sqrt(syst.get("syst_met_px", 0.0) ** 2 + syst.get("syst_met_py", 0.0)),
                            "pairing": pairing,
                            "pairing_unc": pairing_unc,
                            "precision": precision,
                        })
                    except Exception as e:
                        print(f"[WARN] Skipping {syst_dir}: {e}")
        return pd.DataFrame(rows)

    syst_df = collect_results(Path('data/QE/systematics'), ["full", "scratch"])
    # df = collect_results(Path('data/QE/systematics'), ["full"])
    syst_df.sort_values(["model_name", "noise_name"]).reset_index(drop=True)

    grp = syst_df.groupby(["model_name", "noise_name"])

    syst_df["pairing_norm"] = (syst_df["pairing"] - grp["pairing"].transform("mean"))  # / syst_df["pairing_unc"]

    syst_df["precision_norm"] = (syst_df["precision"] - grp["precision"].transform("mean"))

    syst_df["precision_unc"] = 1

    return data, syst_df


def plot_qe_results(
        data,
        systematics_data=None,
        *,
        output_root: str = "plot",
        file_format: str = "pdf",
        dpi: int | None = None,
        save_individual_axes: bool = True,
        with_legend: bool = True,
        style: PlotStyle | None = None,
        fig_scale: float | None = None,
        fig_aspect: float | None = None,
        bar_train_size: int | None = None,
        config: dict | None = None,
):
    """Plot QE figures with optional per-figure font overrides.

    Each plot-specific config (``loss``, ``pair_scatter``, ``pair_bar``,
    ``delta_scatter``, ``delta_bar``) may include a ``style`` entry that
    overrides the base :class:`PlotStyle` so you can tune font sizes or
    scaling for individual figures without affecting the others.
    """
    from plot_styles.style import QE_DATASET_MARKERS, QE_DATASET_PRETTY

    cfg = _merge_configs(DEFAULT_QE_CONFIG, config)

    plot_dir = os.path.join(output_root, "QE")
    base_scale = fig_scale if fig_scale is not None else (style.figure_scale if style else 1.0)

    os.makedirs(plot_dir, exist_ok=True)

    qe_heads = cfg.get("heads", None)
    head_iter = qe_heads if qe_heads else [None]

    loss_outputs = {}
    for head in head_iter:
        head_df = data[data["head"] == head] if head is not None and "head" in data else data
        loss_style = _resolve_style(style, cfg["loss"].get("style"))
        loss_scale = fig_scale if fig_scale is not None else (loss_style.figure_scale if loss_style else base_scale)

        fig, axes, active_models = plot_loss(
            head_df,
            train_sizes=cfg["train_sizes"],
            model_order=cfg["models"],
            dataset_markers=QE_DATASET_MARKERS,
            dataset_pretty=QE_DATASET_PRETTY,
            **cfg["loss"],
            fig_scale=loss_scale,
            fig_aspect=fig_aspect,
        )

        if with_legend:
            plot_legend(
                fig,
                active_models=active_models,
                train_sizes=cfg["train_sizes"],
                dataset_markers=QE_DATASET_MARKERS,
                dataset_pretty=QE_DATASET_PRETTY,
                model_colors=MODEL_COLORS,
                head_order=[head] if head is not None else None,
                head_linestyles=HEAD_LINESTYLES,
                **cfg.get("legend", {}),
                style=loss_style,
            )

        fig.savefig(
            os.path.join(plot_dir, _with_ext(f"loss_{head if head is not None else 'overall'}", file_format)),
            bbox_inches="tight",
            **_save_kwargs(file_format, dpi),
        )
        loss_outputs[head if head is not None else "overall"] = {
            "fig": fig,
            "axes": axes,
            "active_models": active_models,
        }

    pair_bar_size = bar_train_size or max(cfg["train_sizes"])
    pair_scatter_cfg = cfg["pair_scatter"]
    pair_scatter_style = _resolve_style(style, pair_scatter_cfg.get("style"))
    pair_scatter_scale = fig_scale if fig_scale is not None else (
        pair_scatter_style.figure_scale if pair_scatter_style else base_scale)
    fig_pair_scatter, ax_pair_scatter, active_pair = plot_metric_scatter(
        data,
        metric=pair_scatter_cfg.get("metric", "pairing"),
        model_order=cfg["models"],
        train_sizes=cfg["train_sizes"],
        dataset_markers=QE_DATASET_MARKERS,
        dataset_pretty=QE_DATASET_PRETTY,
        head_order=qe_heads if qe_heads else None,
        **{k: v for k, v in pair_scatter_cfg.items() if k not in {"metric"}},
        fig_scale=pair_scatter_scale,
        fig_aspect=fig_aspect,
    )
    if with_legend:
        plot_legend(
            fig_pair_scatter,
            active_models=active_pair,
            train_sizes=cfg["train_sizes"],
            dataset_markers=QE_DATASET_MARKERS,
            dataset_pretty=QE_DATASET_PRETTY,
            model_colors=MODEL_COLORS,
            head_order=qe_heads if qe_heads else None,
            head_linestyles=HEAD_LINESTYLES,
            **cfg.get("legend", {}),
            style=pair_scatter_style,
        )
    fig_pair_scatter.savefig(
        os.path.join(plot_dir, _with_ext("pair_scatter", file_format)),
        bbox_inches="tight",
        **_save_kwargs(file_format, dpi),
    )

    pair_bar_style = _resolve_style(style, cfg["pair_bar"].get("style"))
    pair_bar_scale = fig_scale if fig_scale is not None else (
        pair_bar_style.figure_scale if pair_bar_style else base_scale)
    fig_pair_bar, ax_pair_bar, _ = plot_metric_bar(
        data,
        metric=cfg["pair_bar"].get("metric", "pairing"),
        model_order=cfg["models"],
        head_order=qe_heads if qe_heads else [None],
        train_size_for_bar=pair_bar_size,
        **{k: v for k, v in cfg["pair_bar"].items() if k not in {"metric"}},
        fig_scale=pair_bar_scale,
        fig_aspect=fig_aspect,
    )
    fig_pair_bar.savefig(
        os.path.join(plot_dir, _with_ext("pair_bar", file_format)),
        bbox_inches="tight",
        **_save_kwargs(file_format, dpi),
    )

    delta_scatter_cfg = cfg["delta_scatter"]
    delta_scatter_style = _resolve_style(style, delta_scatter_cfg.get("style"))
    delta_scatter_scale = fig_scale if fig_scale is not None else (
        delta_scatter_style.figure_scale if delta_scatter_style else base_scale)
    fig_delta_scatter, ax_delta_scatter, active_delta = plot_metric_scatter(
        data,
        metric=delta_scatter_cfg.get("metric", "deltaD"),
        model_order=cfg["models"],
        train_sizes=cfg["train_sizes"],
        dataset_markers=QE_DATASET_MARKERS,
        dataset_pretty=QE_DATASET_PRETTY,
        head_order=qe_heads if qe_heads else None,
        **{k: v for k, v in delta_scatter_cfg.items() if k not in {"metric"}},
        fig_scale=delta_scatter_scale,
        fig_aspect=fig_aspect,
    )
    if with_legend:
        plot_legend(
            fig_delta_scatter,
            active_models=active_delta,
            train_sizes=cfg["train_sizes"],
            dataset_markers=QE_DATASET_MARKERS,
            dataset_pretty=QE_DATASET_PRETTY,
            model_colors=MODEL_COLORS,
            head_order=qe_heads if qe_heads else None,
            head_linestyles=HEAD_LINESTYLES,
            **cfg.get("legend", {}),
            style=delta_scatter_style,
        )
    fig_delta_scatter.savefig(
        os.path.join(plot_dir, _with_ext("deltaD_scatter", file_format)),
        bbox_inches="tight",
        **_save_kwargs(file_format, dpi),
    )

    delta_bar_style = _resolve_style(style, cfg["delta_bar"].get("style"))
    delta_bar_scale = fig_scale if fig_scale is not None else (
        delta_bar_style.figure_scale if delta_bar_style else base_scale)
    fig_delta_bar, ax_delta_bar, _ = plot_metric_bar(
        data,
        metric=cfg["delta_bar"].get("metric", "deltaD"),
        model_order=cfg["models"],
        head_order=qe_heads if qe_heads else [None],
        train_size_for_bar=pair_bar_size,
        **{k: v for k, v in cfg["delta_bar"].items() if k not in {"metric"}},
        fig_scale=delta_bar_scale,
        fig_aspect=fig_aspect,
    )
    fig_delta_bar.savefig(
        os.path.join(plot_dir, _with_ext("deltaD_bar", file_format)),
        bbox_inches="tight",
        **_save_kwargs(file_format, dpi),
    )

    systematic_output = {}
    if systematics_data is not None and not systematics_data.empty:
        syst_cases = _prepare_systematic_cases(cfg["systematics"], systematics_data)
        for case in syst_cases:
            syst_cfg = case["config"]
            syst_cfg.update({k: v for k, v in case['_EXTRA'].items() if k not in {'name', 'label', 'color_col'}})
            syst_style = _resolve_style(style, syst_cfg.get("style"))
            syst_scale = fig_scale if fig_scale is not None else (
                syst_style.figure_scale if syst_style else base_scale)
            fig_syst, ax_syst, active_syst = plot_systematic_scatter(
                case["data"],
                model_order=syst_cfg.get("models", cfg["models"]),
                metric_col=syst_cfg.get("metric_col", "pairing_norm"),
                unc_col=syst_cfg.get("unc_col", "pairing_unc"),
                color_col=case.get("color_col", syst_cfg.get("color_col", "syst_jes")),
                **{k: v for k, v in syst_cfg.items() if
                   k not in {"style", "models", "metric_col", "unc_col", "color_col", "suffix", "noise_names"}},
                style=syst_style,
                fig_scale=syst_scale,
                fig_aspect=fig_aspect,
            )
            suffix = case.get("suffix") or syst_cfg.get("suffix", case["metric_name"])
            fig_syst.savefig(
                os.path.join(plot_dir, _with_ext(f"systematics_{suffix}_scatter", file_format)),
                bbox_inches="tight",
                **_save_kwargs(file_format, dpi),
            )
            systematic_output[suffix] = {
                "fig": fig_syst,
                "axes": [ax_syst],
                "active_models": active_syst,
                "metric_name": case["metric_name"],
                "noise_name": case.get("noise_name"),
            }

    plot_task_legend(
        plot_dir=plot_dir,
        model_order=cfg["models"],
        train_sizes=cfg["train_sizes"],
        dataset_markers=QE_DATASET_MARKERS,
        dataset_pretty=QE_DATASET_PRETTY,
        head_order=qe_heads if qe_heads else None,
        legends=["dataset", "heads", "models"],
        file_format=file_format,
        dpi=dpi,
        fig_scale=fig_scale,
        fig_aspect=fig_aspect,
        **{k: v for k, v in cfg["legend"].items() if k not in {"legends"}},
    )

    return {
        "loss": loss_outputs,
        "pair": {
            "scatter": {"fig": fig_pair_scatter, "axes": [ax_pair_scatter], "active_models": active_pair},
            "bar": {"fig": fig_pair_bar, "axes": [ax_pair_bar], "active_models": active_pair},
        },
        "delta": {
            "scatter": {"fig": fig_delta_scatter, "axes": [ax_delta_scatter], "active_models": active_delta},
            "bar": {"fig": fig_delta_bar, "axes": [ax_delta_bar], "active_models": active_delta},
        },
        "systematics": systematic_output,
    }


def read_bsm_data(folder_path):
    pattern = re.compile(
        r"^(?P<model>.+?)"
        r"-assignment-(?P<assignment>on|off)"
        r"(?:-segmentation-(?P<segmentation>on|off))?"  # <-- OPTIONAL
        r"-dataset_size(?P<size>[\d.]+)$"
    )

    MODEL_NAME_MAPPING = {
        "evenet-pretrainv1": "Ablation",
        "evenet-pretrain-ablation1": "SSL",
        "evenet-pretrain-ablation4": "Nominal",
        "evenet-scratch": "Scratch",
        "spanet-scratch": "SPANet",
    }

    HEAD_NAME_MAPPING = {
        (False, False): "Cls",
        (False, True): "Cls+Seg",
        (True, False): "Cls+Asn",
        (True, True): "Cls+Asn+Seg",
    }

    def get_name(folder_name):
        m = pattern.match(folder_name)
        if not m:
            print(f"Folder name {folder_name} does not match the expected pattern.")
            return None

        if m.group("model") not in MODEL_NAME_MAPPING:
            print(f"Model {m.group('model')} not found in folder {folder_name}")
            return None

        params = m.groupdict()
        row = {
            "model": MODEL_NAME_MAPPING[params["model"]],
            "raw_model": params["model"],
            "head": HEAD_NAME_MAPPING[(params["assignment"] == "on", params["segmentation"] == "on")],
            "raw_dataset_size": params["size"],
            "train_size": int(float(params["size"]) * total_N),
        }

        return row

    total_N = 1_000  # in thousands

    # -- Loss Summary -- #
    loss_rows = []
    root = os.path.join(folder_path, "loss_summary")
    epoch_file = os.path.join(folder_path, "loss_epoch.json")
    if os.path.isfile(epoch_file):
        with open(epoch_file) as f:
            epoch_data = json.load(f)
    for dir_path, folders, _ in os.walk(root):
        for folder in folders:
            with open(os.path.join(dir_path, folder, "loss.json")) as f:
                loss_data = json.load(f)

            folder_info = get_name(folder)
            if folder_info is None:
                continue
            for mass, heads in loss_data.items():
                row = {
                    "mass_a": mass,
                    **folder_info,
                    **{
                        (f"{h}_loss" if "loss" not in h else "val_loss"): v
                        for h, v in heads.items()
                    }
                }
                assign_flag = "Assign" in row["head"]
                seg_flag = "Seg" in row["head"]
                key = (
                    f"evenet-ma{row['mass_a']}-"
                    f"{row['raw_model'].replace('evenet-', '')}-"
                    f"{row['raw_dataset_size']}-"
                    f"{assign_flag}-{seg_flag}"
                )
                row["effective_step"] = epoch_data.get(key, {}).get("effective_steps", None)
                loss_rows.append(row)
    df_loss = pd.DataFrame(loss_rows)

    # -- SIC Summary -- #
    sic_rows = []
    root = os.path.join(folder_path, "fit")
    for dir_path, folders, _ in os.walk(root):
        for folder in folders:
            roc_path = os.path.join(dir_path, folder, "summary", "roc_results.npz")
            if not os.path.isfile(roc_path):
                continue

            folder_info = get_name(folder)
            if folder_info is None:
                continue

            roc_npz = np.load(roc_path, allow_pickle=True)

            for mass in ["30", "40", "60"]:
                key = f"haa_ma{mass}"
                if key not in roc_npz:
                    continue

                try:
                    sr = roc_npz[key].item()["SR"]
                except Exception:
                    continue

                if "TPR" not in sr or "FPR" not in sr:
                    continue

                TPR = np.asarray(sr["TPR"])[2:]
                FPR = np.asarray(sr["FPR"])[2:]
                FPR_unc = np.asarray(sr.get("FPR-unc", np.zeros_like(FPR)))[2:]

                row = {
                    "mass_a": mass,
                    "TPR": TPR,
                    "FPR": FPR,
                    "FPR_unc": FPR_unc,
                    **folder_info,
                }
                sic_rows.append(row)
    df_sic = pd.DataFrame(sic_rows)

    # -- Pairing Suumary -- #
    pairing_rows = []
    root = os.path.join(folder_path, "assignment_metrics")
    for dir_path, folders, _ in os.walk(root):
        for folder in folders:
            pairing_path = os.path.join(dir_path, folder, "summary.json")
            if not os.path.isfile(pairing_path):
                continue

            folder_info = get_name(folder)
            if folder_info is None:
                continue

            with open(pairing_path, 'r') as f:
                pairing_data = json.load(f)

            for mass, metrics in pairing_data.items():
                row = {
                    "mass_a": mass.replace("haa_ma", ""),
                    **folder_info,
                    **{
                        "pairing" if 'unc' not in k else "pairing_unc": v * 100 for k, v in metrics.items()
                        if "*a" in k and "event_purity" in k
                    }
                }
                pairing_rows.append(row)
    df_pairing = pd.DataFrame(pairing_rows)

    # Merge loss and sic dataframes according to model, assignment, segmentation, train_size, mass_a
    dfs = [df_loss, df_sic, df_pairing]
    keys = ["model", "head", "train_size", "mass_a"]

    df = reduce(lambda left, right: pd.merge(left, right, on=keys, how="outer"), dfs)

    # -- Systematics Summary -- #
    systematics_path = os.path.join(folder_path, "jse_summary.json")
    syst_df = pd.DataFrame()
    if os.path.isfile(systematics_path):
        with open(systematics_path) as f:
            syst_data = json.load(f)

        model_name_map = {
            "scratch": "Scratch",
            "pretrain-ablation1": "SSL",
            "pretrain-ablation4": "Nominal",
        }

        syst_df = pd.DataFrame(syst_data)
        syst_df["model_pretty"] = syst_df["model"].map(model_name_map).fillna(syst_df["model"])
        syst_df["jes_shift_percent"] = syst_df["noise"] * 100
        syst_df["pairing_eff_percent"] = syst_df["pairing_eff"] * 100
        syst_df["pairing_eff_unc_percent"] = syst_df["pairing_eff_unc"] * 100

        syst_df["sic_norm"] = (syst_df["sic_max"] - syst_df.groupby("model_pretty")["sic_max"].transform("mean")) \
                              / syst_df["sic_unc"]
        syst_df["pairing_norm"] = (syst_df["pairing_eff_percent"] - syst_df.groupby("model_pretty")[
            "pairing_eff_percent"].transform("mean"))  # / syst_df["pairing_eff_unc_percent"]

    return df, syst_df


def plot_bsm_results(
        data,
        systematics_data=None,
        *,
        output_root: str = "plot",
        file_format: str = "pdf",
        dpi: int | None = None,
        save_individual_axes: bool = True,
        with_legend: bool = True,
        style: PlotStyle | None = None,
        fig_scale: float | None = None,
        fig_aspect: float | None = None,
        bar_train_size: int | None = None,
        config: dict | None = None,
):
    """Plot BSM figures with optional per-figure font overrides.

    Each plot config (``loss``, ``pair_scatter``, ``pair_bar``, ``sic``)
    accepts a ``style`` entry to override the base :class:`PlotStyle` for
    that specific figure, enabling fine-grained font and scaling control.
    """
    from plot_styles.style import BSM_DATASET_MARKERS, BSM_DATASET_PRETTY

    cfg = _merge_configs(DEFAULT_BSM_CONFIG, config)

    plot_dir = os.path.join(output_root, "BSM")
    base_scale = fig_scale if fig_scale is not None else (style.figure_scale if style else 1.0)

    os.makedirs(plot_dir, exist_ok=True)

    bsm_heads = cfg["heads"]
    head_iter = bsm_heads if bsm_heads else [None]

    loss_outputs = {}
    for head in head_iter:
        head_df = data[(data["mass_a"] == "30") & (data["head"] == head)] if head is not None else data[
            data["mass_a"] == "30"]
        loss_style = _resolve_style(style, cfg["loss"].get("style"))
        loss_scale = fig_scale if fig_scale is not None else (loss_style.figure_scale if loss_style else base_scale)

        fig, axes, active_models = plot_loss(
            head_df,
            train_sizes=cfg["train_sizes"],
            model_order=cfg["models"],
            dataset_markers=BSM_DATASET_MARKERS,
            dataset_pretty=BSM_DATASET_PRETTY,
            **cfg["loss"],
            fig_scale=loss_scale,
            fig_aspect=fig_aspect,
        )
        if with_legend:
            plot_legend(
                fig,
                active_models=active_models,
                train_sizes=cfg["train_sizes"],
                dataset_markers=BSM_DATASET_MARKERS,
                dataset_pretty=BSM_DATASET_PRETTY,
                model_colors=MODEL_COLORS,
                head_order=[head] if head is not None else None,
                head_linestyles=HEAD_LINESTYLES,
                **cfg.get("legend", {}),
                style=loss_style,
            )
        fig.savefig(
            os.path.join(plot_dir, _with_ext(f"loss_{head if head is not None else 'overall'}", file_format)),
            bbox_inches="tight",
            **_save_kwargs(file_format, dpi),
        )
        loss_outputs[head if head is not None else "overall"] = {
            "fig": fig,
            "axes": axes,
            "active_models": active_models,
        }

    pair_bar_size = bar_train_size or cfg["typical_dataset_size"]
    pair_scatter_cfg = cfg["pair_scatter"]
    pair_scatter_style = _resolve_style(style, pair_scatter_cfg.get("style"))
    pair_scatter_scale = fig_scale if fig_scale is not None else (
        pair_scatter_style.figure_scale if pair_scatter_style else base_scale)
    fig_pair_scatter, ax_pair_scatter, active_pair = plot_metric_scatter(
        data[data["mass_a"] == "30"],
        metric=pair_scatter_cfg.get("metric", "pairing"),
        model_order=cfg["models"],
        train_sizes=cfg["train_sizes"],
        dataset_markers=BSM_DATASET_MARKERS,
        dataset_pretty=BSM_DATASET_PRETTY,
        head_order=bsm_heads if bsm_heads else None,
        **{k: v for k, v in pair_scatter_cfg.items() if k not in {"metric"}},
        fig_scale=pair_scatter_scale,
        fig_aspect=fig_aspect,
    )
    if with_legend:
        plot_legend(
            fig_pair_scatter,
            active_models=active_pair,
            train_sizes=cfg["train_sizes"],
            dataset_markers=BSM_DATASET_MARKERS,
            dataset_pretty=BSM_DATASET_PRETTY,
            model_colors=MODEL_COLORS,
            head_order=bsm_heads if bsm_heads else None,
            head_linestyles=HEAD_LINESTYLES,
            **cfg.get("legend", {}),
            style=pair_scatter_style,
        )
    fig_pair_scatter.savefig(
        os.path.join(plot_dir, _with_ext("pair_scatter", file_format)),
        bbox_inches="tight",
        **_save_kwargs(file_format, dpi),
    )

    pair_bar_style = _resolve_style(style, cfg["pair_bar"].get("style"))
    pair_bar_scale = fig_scale if fig_scale is not None else (
        pair_bar_style.figure_scale if pair_bar_style else base_scale)
    fig_pair_bar, ax_pair_bar, _ = plot_metric_bar(
        data[data["mass_a"] == "30"],
        metric=cfg["pair_bar"].get("metric", "pairing"),
        model_order=cfg["models"],
        head_order=cfg['pair_heads'] if cfg['pair_heads'] else [None],
        train_size_for_bar=pair_bar_size,
        **{k: v for k, v in cfg["pair_bar"].items() if k not in {"metric"}},
        fig_scale=pair_bar_scale,
        fig_aspect=fig_aspect,
    )
    fig_pair_bar.savefig(
        os.path.join(plot_dir, _with_ext("pair_bar", file_format)),
        bbox_inches="tight",
        **_save_kwargs(file_format, dpi),
    )

    sic_cfg = cfg["sic"]
    sic_style = _resolve_style(style, sic_cfg.get("style"))
    sic_scale = fig_scale if fig_scale is not None else (sic_style.figure_scale if sic_style else base_scale)
    sic_figs, active_sic = sic_plot_individual(
        data[data["mass_a"] == "30"],
        model_order=cfg["models"],
        train_sizes=cfg["train_sizes"],
        head_order=bsm_heads if bsm_heads else None,
        dataset_markers=BSM_DATASET_MARKERS,
        dataset_pretty=BSM_DATASET_PRETTY,
        x_indicator=sic_cfg.get("x_indicator", cfg["typical_dataset_size"]),
        x_indicator_text_config=sic_cfg.get("x_indicator_text_config", None),
        y_min=sic_cfg.get("y_min", [0, 0, 0.85]),
        y_max=sic_cfg.get("y_max", None),
        fig_scale=sic_scale,
        fig_aspect=fig_aspect,
        fig_size_curve=sic_cfg.get("fig_size_curve", None),
        fig_size_bar=sic_cfg.get("fig_size_bar", None),
        fig_size_scatter=sic_cfg.get("fig_size_scatter", None),
        bar_style=sic_cfg.get("bar_style", sic_style),
        scatter_style=sic_cfg.get("scatter_style", sic_style),
        curve_style=sic_cfg.get("curve_style", sic_style),
    )
    for name, (fig_sic, _) in sic_figs.items():
        fig_sic.savefig(
            os.path.join(plot_dir, _with_ext(f"sic_{name}", file_format)),
            bbox_inches="tight",
            **_save_kwargs(file_format, dpi),
        )

    systematic_output = {}
    if systematics_data is not None and not systematics_data.empty:
        syst_cases = _prepare_systematic_cases(cfg["systematics"], systematics_data)
        for case in syst_cases:
            syst_cfg = case["config"]
            syst_style = _resolve_style(style, syst_cfg.get("style"))
            syst_scale = fig_scale if fig_scale is not None else (syst_style.figure_scale if syst_style else base_scale)
            fig_syst, ax_syst, active_syst = plot_systematic_scatter(
                case["data"],
                model_order=syst_cfg.get("models", cfg["models"]),
                metric_col=syst_cfg.get("metric_col", "sic_norm"),
                unc_col=syst_cfg.get("unc_col", "sic_unc"),
                color_col=case.get("color_col", syst_cfg.get("color_col", "jes_shift_percent")),
                **{k: v for k, v in syst_cfg.items() if
                   k not in {"style", "models", "metric_col", "unc_col", "color_col", "suffix", "noise_names"}},
                style=syst_style,
                fig_scale=syst_scale,
                fig_aspect=fig_aspect,
            )
            suffix = case.get("suffix") or syst_cfg.get("suffix", case["metric_name"])
            fig_syst.savefig(
                os.path.join(plot_dir, _with_ext(f"systematics_{suffix}_scatter", file_format)),
                bbox_inches="tight",
                **_save_kwargs(file_format, dpi),
            )
            systematic_output[suffix] = {
                "fig": fig_syst,
                "axes": [ax_syst],
                "active_models": active_syst,
                "metric_name": case["metric_name"],
                "noise_name": case.get("noise_name"),
            }

    plot_task_legend(
        plot_dir=plot_dir,
        model_order=cfg["models"],
        train_sizes=cfg["train_sizes"],
        dataset_markers=BSM_DATASET_MARKERS,
        dataset_pretty=BSM_DATASET_PRETTY,
        head_order=bsm_heads if bsm_heads else None,
        legends=["dataset", "heads", "models"],
        file_format=file_format,
        dpi=dpi,
        fig_scale=fig_scale,
        fig_aspect=fig_aspect,
        **{k: v for k, v in cfg["legend"].items() if k not in {"legends"}},
    )

    return {
        "loss": loss_outputs,
        "pair": {
            "scatter": {"fig": fig_pair_scatter, "axes": [ax_pair_scatter], "active_models": active_pair},
            "bar": {"fig": fig_pair_bar, "axes": [ax_pair_bar], "active_models": active_pair},
        },
        "sic": {
            name: {"fig": fig_ax[0], "axes": [fig_ax[1]], "active_models": active_sic}
            for name, fig_ax in sic_figs.items()
        },
        "systematics": systematic_output,
    }


def plot_qe_results_webpage(
        data,
        systematics_data=None,
        *,
        output_root: str = "plot",
        file_format: str = "pdf",
        dpi: int | None = None,
        style: PlotStyle | None = None,
        fig_scale: float | None = None,
        fig_aspect: float | None = None,
):
    from plot_styles.style import QE_DATASET_MARKERS, QE_DATASET_PRETTY

    QE_CONFIG = {
        "models": ["Nominal", "Scratch", "SSL", "Ref."],
        # Set to an empty list if heads are not applicable for QE.
        "heads": ["Cls", "Cls+Asn"],
        "train_sizes": [15, 148, 1475, 2950],
        "systematics": DEFAULT_QE_CONFIG.get("systematics", {}),
    }

    qe_heads = QE_CONFIG["heads"]
    loss_names = qe_heads if qe_heads else ["overall"]

    plot_dir = os.path.join(output_root, "QE")

    legend_path = plot_task_legend(
        plot_dir=plot_dir,
        model_order=QE_CONFIG["models"],
        train_sizes=QE_CONFIG["train_sizes"],
        dataset_markers=QE_DATASET_MARKERS,
        dataset_pretty=QE_DATASET_PRETTY,
        head_order=qe_heads if qe_heads else None,
        legends=["dataset", "heads", "models"],
        file_format=file_format,
        dpi=dpi,
        style=style,
        fig_scale=fig_scale,
        fig_aspect=fig_aspect,
    )

    results = plot_qe_results(
        data,
        systematics_data=systematics_data,
        output_root=output_root,
        file_format=file_format,
        dpi=dpi,
        save_individual_axes=True,
        with_legend=False,
        style=style,
        fig_scale=fig_scale,
        fig_aspect=fig_aspect,
    )

    systematics_paths = []
    if results.get("systematics"):
        syst_cases = _prepare_systematic_cases(QE_CONFIG["systematics"], systematics_data)
        for case in syst_cases:
            suffix = case.get("suffix") or case.get("metric_name")
            if suffix not in results["systematics"]:
                continue
            systematics_paths.append(
                os.path.join(plot_dir, _with_ext(f"systematics_{suffix}_scatter", file_format))
            )

    return {
        "legend": legend_path,
        "loss": [os.path.join(plot_dir, _with_ext(f"loss_{head}", file_format)) for head in loss_names],
        "pair": [
            os.path.join(plot_dir, _with_ext("pair_scatter", file_format)),
            os.path.join(plot_dir, _with_ext("pair_bar", file_format)),
        ],
        "delta": [
            os.path.join(plot_dir, _with_ext("deltaD_scatter", file_format)),
            os.path.join(plot_dir, _with_ext("deltaD_bar", file_format)),
        ],
        "systematics": systematics_paths,
    }


def plot_bsm_results_webpage(
        data,
        systematics_data=None,
        *,
        output_root: str = "plot",
        file_format: str = "pdf",
        dpi: int | None = None,
        style: PlotStyle | None = None,
        fig_scale: float | None = None,
        fig_aspect: float | None = None,
):
    from plot_styles.style import BSM_DATASET_MARKERS, BSM_DATASET_PRETTY

    BSM_CONFIG = {
        "models": ["Nominal", "Scratch", "SSL", "SPANet"],
        # Set to an empty list if heads are not applicable for BSM.
        "heads": ["Cls", "Cls+Asn"],
        "train_sizes": [10, 30, 100, 300],
        "systematics": DEFAULT_BSM_CONFIG.get("systematics", {}),
    }

    bsm_heads = BSM_CONFIG["heads"]
    loss_names = bsm_heads if bsm_heads else ["overall"]

    plot_dir = os.path.join(output_root, "BSM")

    legend_path = plot_task_legend(
        plot_dir=plot_dir,
        model_order=BSM_CONFIG["models"],
        train_sizes=BSM_CONFIG["train_sizes"],
        dataset_markers=BSM_DATASET_MARKERS,
        dataset_pretty=BSM_DATASET_PRETTY,
        head_order=bsm_heads if bsm_heads else None,
        legends=["dataset", "heads", "models"],
        file_format=file_format,
        dpi=dpi,
        style=style,
        fig_scale=fig_scale,
        fig_aspect=fig_aspect,
    )

    results = plot_bsm_results(
        data,
        systematics_data=systematics_data,
        output_root=output_root,
        file_format=file_format,
        dpi=dpi,
        save_individual_axes=True,
        with_legend=False,
        style=style,
        fig_scale=fig_scale,
        fig_aspect=fig_aspect,
    )

    systematics_paths = []
    if results.get("systematics"):
        syst_cases = _prepare_systematic_cases(BSM_CONFIG["systematics"], systematics_data)
        for case in syst_cases:
            suffix = case.get("suffix") or case.get("metric_name")
            if suffix not in results["systematics"]:
                continue
            systematics_paths.append(
                os.path.join(plot_dir, _with_ext(f"systematics_{suffix}_scatter", file_format))
            )

    return {
        "legend": legend_path,
        "loss": [os.path.join(plot_dir, _with_ext(f"loss_{head}", file_format)) for head in loss_names],
        "pair": [
            os.path.join(plot_dir, _with_ext("pair_scatter", file_format)),
            os.path.join(plot_dir, _with_ext("pair_bar", file_format)),
        ],
        "sic": [
            os.path.join(plot_dir, _with_ext("sic_curve", file_format)),
            os.path.join(plot_dir, _with_ext("sic_bar", file_format)),
            os.path.join(plot_dir, _with_ext("sic_scatter", file_format)),
        ],
        "systematics": systematics_paths,
    }


def read_ad_data(file_path):
    AD_MODEL_MAPPING = {
        "EveNet-f.t.(SSL)": "SSL",
        "EveNet-f.t.(Cls+Gen)": "Nominal",
        "EveNet-f.t.(Cls+Gen+Assign)": "Ablation",
        "EveNet-scratch": "Scratch",
    }

    ad_sig = os.path.join(file_path, "significance_summary.json")
    if os.path.isfile(ad_sig):
        with open(ad_sig) as f:
            ad_sig_data = json.load(f)

    def compute_channel_significance(model_stats, cl=0.68):
        """
        Compute median and 68% CL per model per channel.
        Returns a DataFrame: model | channel | median | lower | upper
        """
        records = []
        for model, channels in model_stats.items():
            for channel, qvals in channels.items():
                q2 = qvals
                lower_p = (1 - cl) / 2 * 100
                upper_p = (1 + cl) / 2 * 100
                median = np.median(q2)
                lower = np.percentile(q2, lower_p)
                upper = np.percentile(q2, upper_p)

                m = re.match(r"train-(OS|SS)-test-(OS|SS)", channel)
                train_type, test_type = m.groups()

                records.append({
                    "raw_model": model,
                    "model": AD_MODEL_MAPPING.get(model.replace('-calibrated', ''), model),
                    "calibrated": True if "-calibrated" in model.lower() else False,
                    "channel": channel,
                    "train_type": train_type,
                    "test_type": test_type,
                    "median": median,
                    "lower": lower,
                    "upper": upper,
                    "number": len(qvals)
                })
        return pd.DataFrame(records)

    ad_sig_df = compute_channel_significance(ad_sig_data)

    ad_gen = os.path.join(file_path, "boostrap_summary.json")
    if os.path.isfile(ad_gen):
        with open(ad_gen) as f:
            ad_gen_data = json.load(f)

    def compute_channel_generation(model_stats):
        records = []
        for ibootstrap, summary in model_stats.items():
            for label, val in summary.items():
                if "after cut" not in val:
                    continue

                metrics = val["after cut"]
                group = "OS" if label.endswith("OS") else "SS"

                records.append({
                    "raw_model": label,
                    "model": AD_MODEL_MAPPING.get(re.sub(r"(-(calibrated|OS|SS))+$", "", label), label),
                    "calibrated": True if "-calibrated" in label.lower() else False,
                    "train_type": group,
                    "test_type": None,

                    "bootstrap": ibootstrap,
                    "cov": metrics.get("cov", np.nan),
                    "mmd": metrics.get("mmd", np.nan),
                    "efficiency": metrics.get("efficiency", np.nan),
                    "mean_calibration_difference": metrics.get("mean_calibration_difference", np.nan),
                    "mean_mass_difference": metrics.get("mean_mass_difference", np.nan),
                })

        df = pd.DataFrame(records)
        return df

    ad_gen_df = compute_channel_generation(ad_gen_data)

    return {
        'sig': ad_sig_df,
        'gen': ad_gen_df
    }


def plot_ad_results(
        data,
        *,
        output_root: str = "plot",
        file_format: str = "pdf",
        dpi: int | None = None,
        save_individual_axes: bool = True,
        style: PlotStyle | None = None,
        fig_scale: float | None = None,
        fig_aspect: float | None = None,
        config: dict | None = None,
):
    """Plot AD figures with optional per-figure font overrides.

    ``sig``, ``gen_mmd``, and ``gen_calibration`` configs may provide a
    ``style`` entry to override the base :class:`PlotStyle` for
    fine-grained control over fonts or scaling in each plot.
    """
    cfg = _merge_configs(DEFAULT_AD_CONFIG, config)

    plot_dir = os.path.join(output_root, "AD")
    base_scale = fig_scale if fig_scale is not None else (style.figure_scale if style else 1.0)

    sig_style = _resolve_style(style, cfg["sig"].get("style"))
    sig_scale = fig_scale if fig_scale is not None else (sig_style.figure_scale if sig_style else base_scale)

    plot_task_legend(
        plot_dir=plot_dir,
        model_order=cfg["models"],
        train_sizes=None,
        dataset_markers=None,
        dataset_pretty=None,
        head_order=None,
        legends=["models", "calibration"],
        file_format=file_format,
        dpi=dpi,
        style=style,
        fig_scale=fig_scale,
        fig_aspect=fig_aspect,
    )

    plot_ad_sig_summary(
        data['sig'],
        models_order=cfg["models"],
        channels_order=cfg["sig"]["channels_order"],
        show_error=cfg["sig"].get("show_error", True),
        var=cfg["sig"].get("var", "median"),
        f_name=_with_ext(cfg["sig"].get("f_name", "ad_significance"), file_format),
        plot_dir=plot_dir,
        dpi=dpi,
        file_format=file_format,
        y_ref=cfg["sig"].get("y_ref", 6.4),
        style=sig_style,
        fig_scale=sig_scale,
        fig_aspect=fig_aspect,
        fig_size=cfg["sig"].get("fig_size", None),
        include_uncalibrated=cfg["sig"].get("include_uncalibrated", True)
    )

    gen_outputs = []
    for key in ["gen_mmd", "gen_calibration"]:
        gen_cfg = cfg[key]
        gen_style = _resolve_style(style, gen_cfg.get("style"))
        gen_scale = fig_scale if fig_scale is not None else (gen_style.figure_scale if gen_style else base_scale)

        f_name = _with_ext(gen_cfg.get("f_name", key), file_format)
        plot_ad_gen_summary(
            data['gen'],
            models_order=cfg["models"],
            metric=gen_cfg.get("metric", "mmd"),
            label=gen_cfg.get("label", ""),
            train_types=gen_cfg.get("train_types"),
            region_gap=gen_cfg.get("region_gap", 0.4),
            include_uncalibrated=gen_cfg.get("include_uncalibrated"),
            f_name=f_name,
            plot_dir=plot_dir,
            y_min=gen_cfg.get("y_min"),
            percentage=gen_cfg.get("percentage", False),
            dpi=dpi,
            file_format=file_format,
            style=gen_style,
            fig_scale=gen_scale,
            fig_aspect=fig_aspect,
            in_figure=True,
            figsize=gen_cfg.get("fig_size"),
            with_legend=gen_cfg.get("with_legend", True),
        )
        gen_outputs.append(os.path.join(plot_dir, f_name))

    return {
        "legend": None,
        "sig": [os.path.join(plot_dir, _with_ext("ad_significance", file_format))],
        "gen": gen_outputs,
    }


def plot_ad_results_webpage(
        data,
        *,
        output_root: str = "plot",
        file_format: str = "pdf",
        dpi: int | None = None,
        style: PlotStyle | None = None,
        fig_scale: float | None = None,
        fig_aspect: float | None = None,
):
    plot_dir = os.path.join(output_root, "AD")

    outputs = plot_ad_results(
        data,
        output_root=output_root,
        file_format=file_format,
        dpi=dpi,
        save_individual_axes=True,
        style=style,
        fig_scale=fig_scale,
        fig_aspect=fig_aspect,
    )

    return outputs


def _collect_grid_series(
        grid_data: pd.DataFrame,
        *,
        series_specs: list[dict],
        metric_col: str,
        unc_col: str,
):
    selections = []
    for spec in series_specs:
        mask = grid_data["model"].eq(spec["model"])
        if spec.get("type") is not None:
            mask &= grid_data["type"].eq(spec["type"])
        selections.append(mask)

    if not selections:
        return [], {}, {}, [], {}

    selected = grid_data[np.logical_or.reduce(selections)]
    if selected.empty:
        return [], {}, {}, [], {}

    points = sorted({(row.m_X, row.m_Y) for row in selected.itertuples(index=False)})
    sic_by_model = {}
    unc_by_model = {}
    model_order = []
    color_map = {}

    for spec in series_specs:
        label = spec.get("label", spec["model"])
        model_order.append(label)
        if "color" in spec:
            color_map[label] = spec["color"]

        df_model = grid_data[grid_data["model"].eq(spec["model"])]
        if spec.get("type") is not None:
            df_model = df_model[df_model["type"].eq(spec["type"])]

        grouped_metric = df_model.groupby(["m_X", "m_Y"])[metric_col].mean()
        grouped_unc = (
            df_model.groupby(["m_X", "m_Y"])[unc_col].mean()
            if unc_col in df_model
            else pd.Series(dtype=float)
        )
        metric_map = {key: value for key, value in grouped_metric.items()}
        unc_map = {key: value for key, value in grouped_unc.items()}

        sic_by_model[label] = {point: metric_map.get(point, np.nan) for point in points}
        unc_by_model[label] = {point: unc_map.get(point, np.nan) for point in points}

    return points, sic_by_model, unc_by_model, model_order, color_map


def _collect_grid_metric_map(
        grid_data: pd.DataFrame,
        *,
        series_specs: list[dict],
        metric_col: str,
        points: list[tuple[float, float]],
):
    metric_by_model = {}
    for spec in series_specs:
        label = spec.get("label", spec["model"])
        df_model = grid_data[grid_data["model"].eq(spec["model"])]
        if spec.get("type") is not None:
            df_model = df_model[df_model["type"].eq(spec["type"])]

        if metric_col in df_model.columns:
            grouped_metric = df_model.groupby(["m_X", "m_Y"])[metric_col].mean()
        else:
            grouped_metric = pd.Series(dtype=float)
        metric_map = {key: value for key, value in grouped_metric.items()}
        metric_by_model[label] = {point: metric_map.get(point, np.nan) for point in points}

    return metric_by_model


def plot_grid_results(
        grid_data: pd.DataFrame,
        cutflow_df: pd.DataFrame | None = None,
        *,
        output_root: str = "plot",
        file_format: str = "pdf",
        dpi: int | None = None,
        style: PlotStyle | None = None,
        config: dict | None = None,
):
    if isinstance(grid_data, tuple):
        grid_data, cutflow_df = grid_data
    cfg = _merge_configs(DEFAULT_GRID_CONFIG, config)
    plot_dir = os.path.join(output_root, cfg.get("output_subdir", "Grid"))
    os.makedirs(plot_dir, exist_ok=True)

    outputs = {}
    for name, plot_cfg in cfg.get("plots", {}).items():
        points, sic_by_model, unc_by_model, model_order, color_map = _collect_grid_series(
            grid_data,
            series_specs=plot_cfg.get("series", []),
            metric_col=cfg.get("metric_col", "max_sic"),
            unc_col=cfg.get("unc_col", "max_sic_unc"),
        )
        if not points:
            continue

        plot_config = _merge_configs(cfg.get("base_plot_config", {}), plot_cfg.get("plot_config", {}))
        plot_config["models"] = model_order
        if color_map:
            plot_config["color_map"] = color_map

        unc_cfg = dict(plot_config.get("unc", {}))
        if unc_cfg.get("enabled", True) and not unc_cfg.get("models"):
            unc_cfg["models"] = model_order
        plot_config["unc"] = unc_cfg
        plot_config['raw_model_series'] = [f"{m['model']}_{m['type']}" for m in plot_cfg.get("series", [])]

        aux_cfg = plot_config.get("aux_panel", {})
        aux_by_model = None
        if aux_cfg.get("enabled", False):
            aux_by_model = _collect_grid_metric_map(
                grid_data,
                series_specs=plot_cfg.get("series", []),
                metric_col=aux_cfg.get("metric_col", "effective_steps"),
                points=points,
            )

        with use_style(style):
            fig, axes = plot_unrolled_grid_with_winner_and_ratios(
                points,
                sic_by_model,
                unc_by_model=unc_by_model,
                aux_by_model=aux_by_model,
                cutflow_df=cutflow_df,
                config=plot_config,
                style=style,
            )

        output_name = plot_cfg.get("output_name", f"grid_{name}")
        output_path = os.path.join(plot_dir, _with_ext(output_name, file_format))
        fig.savefig(output_path, **_save_kwargs(file_format, dpi))
        outputs[name] = {
            "fig": fig,
            "axes": axes,
            "path": output_path,
            "title": plot_cfg.get("title", name),
        }

    loss_cfg = cfg.get("loss")
    if loss_cfg and loss_cfg.get("enabled", True):
        loss_style = _resolve_style(style, loss_cfg.get("style"))
        loss_scale = loss_cfg.get(
            "fig_scale",
            loss_style.figure_scale if loss_style and loss_style.figure_scale else 1.0,
        )
        fig, ax = plot_grid_loss(
            grid_data,
            model_specs=loss_cfg.get("models", []),
            loss_col=loss_cfg.get("loss_col", "min_val_loss"),
            raw_step_col=loss_cfg.get("raw_step_col", "effective_steps_raw"),
            per_signal_step_col=loss_cfg.get("per_signal_step_col", "effective_steps"),
            x_label=loss_cfg.get("x_label", "Effective steps [K]"),
            y_label=loss_cfg.get("y_label", "Min val loss"),
            xscale=loss_cfg.get("xscale", "log"),
            yscale=loss_cfg.get("yscale"),
            y_min=loss_cfg.get("y_min"),
            y_max=loss_cfg.get("y_max"),
            fig_size=loss_cfg.get("fig_size", (6, 6)),
            fig_scale=loss_scale,
            fig_aspect=loss_cfg.get("fig_aspect"),
            grid=loss_cfg.get("grid", False),
            individual_style=loss_cfg.get("individual"),
            param_points=loss_cfg.get("param_points"),
            param_style=loss_cfg.get("param"),
            density=loss_cfg.get("density"),
            legend_config=loss_cfg.get("legend"),
            style=loss_style,
        )

        output_name = loss_cfg.get("output_name", "grid_loss")
        output_path = os.path.join(plot_dir, _with_ext(output_name, file_format))
        fig.savefig(output_path, **_save_kwargs(file_format, dpi))
        outputs["loss"] = {
            "fig": fig,
            "axes": [ax],
            "path": output_path,
            "title": loss_cfg.get("title", "Grid loss"),
        }

    return outputs


def plot_grid_results_webpage(
        grid_data: pd.DataFrame,
        cutflow_df: pd.DataFrame | None = None,
        *,
        output_root: str = "plot",
        file_format: str = "pdf",
        dpi: int | None = None,
        style: PlotStyle | None = None,
        config: dict | None = None,
):
    results = plot_grid_results(
        grid_data,
        cutflow_df,
        output_root=output_root,
        file_format=file_format,
        dpi=dpi,
        style=style,
        config=config,
    )

    output_dir = Path(output_root)
    merged_cfg = _merge_configs(DEFAULT_GRID_CONFIG, config)
    plot_dir = output_dir / merged_cfg.get("output_subdir", "Grid")

    outputs = {
        name: os.path.join(str(plot_dir), _with_ext(cfg.get("output_name", f"grid_{name}"), file_format))
        for name, cfg in merged_cfg.get("plots", {}).items()
        if name in results
    }
    loss_cfg = merged_cfg.get("loss", {})
    if loss_cfg and loss_cfg.get("enabled", True) and "loss" in results:
        outputs["loss"] = os.path.join(
            str(plot_dir),
            _with_ext(loss_cfg.get("output_name", "grid_loss"), file_format),
        )
    return outputs


def plot_final_paper_figures(
        *,
        qe_data=None,
        bsm_data=None,
        ad_data=None,
        grid_data=None,
        output_root: str = "plot",
        file_format: str = "pdf",
        dpi: int | None = None,
        include_legends: bool = False,
        style: PlotStyle | None = None,
        figure_options: dict | None = None,
        task_configs: dict | None = None,
):
    """High-level entry point for generating the final paper plots.

    This wrapper keeps the plotting surface clean and allows per-figure
    overrides so you can quickly fine-tune font sizes, axis ratios, or
    legend visibility before exporting slides.

    Parameters
    ----------
    qe_data, bsm_data, ad_data, grid_data : DataFrame or None
        Provide the pre-loaded data for each task. Any task set to
        ``None`` is skipped so you can iteratively build a figure set.
    include_legends : bool
        Global toggle for legends. Per-figure overrides via
        ``figure_options`` take precedence. Defaults to ``False`` so the
        exported figures are presentation-ready and uncluttered.
    style : PlotStyle | None
        Baseline style applied to every figure. You can override any
        ``PlotStyle`` field (font sizes, scaling, etc.) per figure by
        supplying a mapping under ``figure_options``.
    figure_options : dict | None
        Optional mapping keyed by ``"qe"``, ``"bsm"``, ``"ad"``, or ``"grid"``.
        Each entry may include any keyword accepted by the respective
        plotting helper (e.g., ``fig_scale``, ``fig_aspect``,
        ``bar_train_size``) plus a ``style`` override.
    """

    figure_options = figure_options or {}
    task_configs = task_configs or {}
    results = {}

    if qe_data is not None:
        if isinstance(qe_data, tuple):
            qe_sum, qe_syst = qe_data
        else:
            qe_sum, qe_syst = qe_data, None

        opts = figure_options.get("qe", {})
        results["qe"] = plot_qe_results(
            qe_sum,
            systematics_data=qe_syst,
            output_root=opts.get("output_root", output_root),
            file_format=opts.get("file_format", file_format),
            dpi=opts.get("dpi", dpi),
            save_individual_axes=opts.get("save_individual_axes", True),
            with_legend=opts.get("with_legend", include_legends),
            style=_resolve_style(style, opts.get("style")),
            fig_scale=opts.get("fig_scale"),
            fig_aspect=opts.get("fig_aspect"),
            bar_train_size=opts.get("bar_train_size"),
            config=task_configs.get("qe"),
        )

    if bsm_data is not None:
        if isinstance(bsm_data, tuple):
            bsm_summary, bsm_systematics = bsm_data
        else:
            bsm_summary, bsm_systematics = bsm_data, None
        opts = figure_options.get("bsm", {})
        results["bsm"] = plot_bsm_results(
            bsm_summary,
            systematics_data=bsm_systematics,
            output_root=opts.get("output_root", output_root),
            file_format=opts.get("file_format", file_format),
            dpi=opts.get("dpi", dpi),
            save_individual_axes=opts.get("save_individual_axes", True),
            with_legend=opts.get("with_legend", include_legends),
            style=_resolve_style(style, opts.get("style")),
            fig_scale=opts.get("fig_scale"),
            fig_aspect=opts.get("fig_aspect"),
            bar_train_size=opts.get("bar_train_size"),
            config=task_configs.get("bsm"),
        )

    if ad_data is not None:
        opts = figure_options.get("ad", {})
        plot_ad_results(
            ad_data,
            output_root=opts.get("output_root", output_root),
            file_format=opts.get("file_format", file_format),
            dpi=opts.get("dpi", dpi),
            style=_resolve_style(style, opts.get("style")),
            fig_scale=opts.get("fig_scale"),
            fig_aspect=opts.get("fig_aspect"),
            config=task_configs.get("ad"),
        )

    if grid_data is not None:
        opts = figure_options.get("grid", {})
        if isinstance(grid_data, tuple):
            grid_df, cutflow_df = grid_data
        else:
            grid_df, cutflow_df = grid_data, None
        results["grid"] = plot_grid_results(
            grid_df,
            cutflow_df,
            output_root=opts.get("output_root", output_root),
            file_format=opts.get("file_format", file_format),
            dpi=opts.get("dpi", dpi),
            style=_resolve_style(style, opts.get("style")),
            config=task_configs.get("grid"),
        )

    return results


def read_grid_data(file_path):
    method_dir = Path(file_path)
    rows = []

    train_info = {
        # batch size, # of GPU
        ("evenet-pretrain", "individual"): (4096, 1),
        ("evenet-pretrain", "param"): (2048, 2),
        ("evenet-scratch", "individual"): (2048, 1),
        ("evenet-scratch", "param"): (2048, 2),
    }

    with open(method_dir / "all_checkpoints.txt") as f:
        all_checkpoints = [line.strip() for line in f if line.strip()]

    pattern_ckpt = re.compile(
        r"""
        ^(?P<model>evenet-(?:pretrain|scratch))/
        (?P<training>(?:individual|parametrized_reduce_factor_x_1_y_1))/
        (?:
            MX-(?P<mX>[\d.]+)_MY-(?P<mY>[\d.]+)/
            |
            All/
        )
        checkpoints/
        checkpoints-val_loss-epoch(?P<epoch>\d+)-(?P<loss>[\d.]+)\.pt$
        """,
        re.VERBOSE,
    )

    ckpt_result = {}

    for s in all_checkpoints:
        m = pattern_ckpt.match(s)
        if not m:
            continue

        model_name = m.group("model")
        train_type = m.group("training")
        train_type = "param" if train_type == "parametrized_reduce_factor_x_1_y_1" else train_type

        # Handle parametrized "All" case
        mX = float(m.group("mX")) if m.group("mX") else None
        mY = float(m.group("mY")) if m.group("mY") else None

        epoch = int(m.group("epoch"))
        val_loss = float(m.group("loss"))

        key = (model_name, train_type, mX, mY)
        ckpt_result[key] = (epoch, val_loss)

    # Regex for MX / MY in filename
    pattern = re.compile(r"MX-(?P<mx>[\d.]+)_MY-(?P<my>[\d.]+)")

    for model_dir in method_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name  # evenet-pretrain, xgb, ...

        for json_file in model_dir.rglob("eval_metrics_*.json"):
            match = pattern.search(json_file.name)
            if match is None:
                continue

            m_X = float(match.group("mx"))
            m_Y = float(match.group("my"))

            # Determine type
            parts = json_file.parts
            if "individual" in parts:
                run_type = "individual"
                ckpt_info = ckpt_result.get((model_name, run_type, m_X, m_Y), (None, None))
            elif any(p.startswith("parametrized_reduce_factor_x_1_y_1") for p in parts):
                run_type = "param"
                ckpt_info = ckpt_result.get((model_name, run_type, None, None), (None, None))
            else:
                run_type = "unknown"
                ckpt_info = (None, None)
            effective_train_info = train_info.get((model_name, run_type), (0, 0))

            # Read JSON
            with open(json_file) as f:
                metrics = json.load(f)

            rows.append({
                "model": model_name,
                "type": run_type,
                "m_X": m_X,
                "m_Y": m_Y,
                "auc": metrics.get("auc"),
                "max_sic": metrics.get("max_sic"),
                "max_sic_unc": metrics.get("max_sic_unc"),
                "trafo_bin_sig": metrics.get("trafo_bin_sig"),
                "min_val_loss": ckpt_info[1],
                "best_epoch": ckpt_info[0],
                "statistics": 0.0,
                "effective_steps": None,
                "effective_batch_size": effective_train_info[0] * effective_train_info[1],
            })

    grid_df = pd.DataFrame(rows)
    selected_grid_df = grid_df[grid_df['type'].isin(['individual', 'param'])].copy()

    # reading cutflows
    cutflows = json.load(open(method_dir / "cutflow.json"))
    rows = []
    bkgs = []
    for sample, counts in cutflows.items():
        m = pattern.match(sample)
        if not m:
            bkgs.append({
                "sample": sample,
                "passed": counts["passed"] / 2000 if sample != "tt1l" else 500,
                "total": counts["total"] / 2000,
            })
            continue

        rows.append({
            "m_X": float(m.group("mx")),
            "m_Y": float(m.group("my")),
            "passed": counts["passed"] / 2000,
            "total": counts["total"] / 2000,
        })

    cutflow_df = pd.DataFrame(rows).sort_values(["m_X", "m_Y"]).reset_index(drop=True)
    cutflow_bkg_df = pd.DataFrame(bkgs)
    bkg_sum = cutflow_bkg_df['passed'].sum()
    sig_sum = cutflow_df["passed"].sum()
    sig_passed_map = {(r.m_X, r.m_Y): r.passed for r in cutflow_df.itertuples(index=False)}

    # individual: per-point signal + global bkg
    mask_ind = selected_grid_df["type"].eq("individual")
    selected_grid_df.loc[mask_ind, "statistics"] = [
        sig_passed_map.get((mx, my), np.nan) + bkg_sum
        for mx, my in zip(selected_grid_df.loc[mask_ind, "m_X"], selected_grid_df.loc[mask_ind, "m_Y"])
    ]

    # param: global (all signals) + global bkg (same for every row)
    mask_par = selected_grid_df["type"].eq("param")
    selected_grid_df.loc[mask_par, "statistics"] = sig_sum + bkg_sum

    selected_grid_df["effective_steps"] = convert_epochs_to_steps(
        epoch=selected_grid_df["best_epoch"],
        train_size=selected_grid_df["statistics"],
        batch_size_per_GPU=selected_grid_df["effective_batch_size"],
        GPUs=1,
    ) / 1000
    selected_grid_df["effective_steps_raw"] = selected_grid_df["effective_steps"]
    selected_grid_df["effective_steps_per_signal"] = selected_grid_df["effective_steps"]
    selected_grid_df.loc[mask_par, "effective_steps"] /= len(cutflow_df)
    selected_grid_df.loc[mask_par, "effective_steps_per_signal"] /= len(cutflow_df)

    return selected_grid_df, cutflow_df


if __name__ == '__main__':
    # qe_data = read_qe_data('data/QE_results_table.csv')
    # plot_qe_results(qe_data)

    # bsm_summary, bsm_systematics = read_bsm_data('data/BSM')
    # plot_bsm_results(bsm_summary, systematics_data=bsm_systematics)

    # ad_data = read_ad_data("data/AD")
    # plot_ad_results(ad_data)

    grid_data = read_grid_data("data/Grid_Study/method_arxiv")

    pass
