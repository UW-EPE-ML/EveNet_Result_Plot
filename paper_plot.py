import re
import json
import os
import pandas as pd
import numpy as np
from functools import reduce

from plot_styles.high_level.plot_loss import plot_loss
from plot_styles.high_level.plot_bar_line import plot_metric_scatter, plot_metric_bar
from plot_styles.sic import sic_plot_individual
from plot_styles.ad_bar import plot_ad_sig_summary, plot_ad_gen_summary
from plot_styles.core.legend import plot_legend, plot_only_legend
from plot_styles.core.theme import PlotStyle, scaled_fig_size, use_style
from plot_styles.style import MODEL_COLORS, HEAD_LINESTYLES

BITMAP_FORMATS = {"png", "jpg", "jpeg", "tiff", "bmp"}

DEFAULT_LEGEND_STYLE = PlotStyle(legend_size=12.0)
DEFAULT_BAR_STYLE = PlotStyle(tick_label_size=19.0, nbins=2, full_axis=True)
DEFAULT_STYLE = PlotStyle(base_font_size=20.0, tick_label_size=19.0)

DEFAULT_QE_CONFIG = {
    # "train_sizes": [15, 148, 1475, 2950],
    "train_sizes": [15, 148, 1475],
    "models": ["Nominal", "SSL", "Scratch"],
    "heads": [],
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
}

DEFAULT_BSM_CONFIG = {
    "train_sizes": [10, 30, 100, 300],
    "typical_dataset_size": 300,
    "models": ["Nominal", "SSL", "Scratch",  "SPANet"],
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
        "x_indicator": 250,
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
        "x_indicator": 250,
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
}

DEFAULT_AD_CONFIG = {
    "models": ["Nominal", "SSL", "Scratch"],
    "heads": [],
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
        "style": PlotStyle(base_font_size=20.0, tick_label_size=19.0, legend_size=19.0),
        "fig_size": (13,6)
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
        "label": "calibration magnitude [%]",
        "f_name": "ad_generation_calibration",
        "fig_size": (6, 4),
        "y_min": 0,
        "percentage": True,
        "train_types": ["OS", "SS"],
        "region_gap": 0.4,
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


def convert_epochs_to_steps(epoch, train_size, batch_size_per_GPU=1024, GPUs=16):
    effective_step = (epoch * train_size * 1000 / (batch_size_per_GPU * GPUs))
    return effective_step


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

    return data


def plot_qe_results(
        data,
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

    qe_heads = cfg["heads"]
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

    return df


def plot_bsm_results(
        data,
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
    }


def plot_qe_results_webpage(
        data,
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
        output_root=output_root,
        file_format=file_format,
        dpi=dpi,
        save_individual_axes=True,
        with_legend=False,
        style=style,
        fig_scale=fig_scale,
        fig_aspect=fig_aspect,
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
    }


def plot_bsm_results_webpage(
        data,
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
        output_root=output_root,
        file_format=file_format,
        dpi=dpi,
        save_individual_axes=True,
        with_legend=False,
        style=style,
        fig_scale=fig_scale,
        fig_aspect=fig_aspect,
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
        in_figure=True,
        fig_size=cfg["sig"].get("fig_size", None),
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


def plot_final_paper_figures(
        *,
        qe_data=None,
        bsm_data=None,
        ad_data=None,
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
    qe_data, bsm_data, ad_data : DataFrame or None
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
        Optional mapping keyed by ``"qe"``, ``"bsm"``, or ``"ad"``.
        Each entry may include any keyword accepted by the respective
        plotting helper (e.g., ``fig_scale``, ``fig_aspect``,
        ``bar_train_size``) plus a ``style`` override.
    """

    figure_options = figure_options or {}
    task_configs = task_configs or {}
    results = {}

    if qe_data is not None:
        opts = figure_options.get("qe", {})
        results["qe"] = plot_qe_results(
            qe_data,
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
        opts = figure_options.get("bsm", {})
        results["bsm"] = plot_bsm_results(
            bsm_data,
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

    return results


if __name__ == '__main__':
    # qe_data = read_qe_data('data/QE_results_table.csv')
    # plot_qe_results(qe_data)

    bsm_data = read_bsm_data('data/BSM')
    plot_bsm_results(bsm_data)

    # ad_data = read_ad_data("data/AD")
    # plot_ad_results(ad_data)

    pass
