import re
import json
import os
import pandas as pd
import numpy as np
from functools import reduce

from plot_styles.high_level.plot_loss import plot_loss
from plot_styles.high_level.plot_bar_line import plot_bar_line
from plot_styles.sic import sic_plot
from plot_styles.ad_bar import plot_ad_sig_summary, plot_ad_gen_summary
from plot_styles.core.legend import plot_legend, plot_only_legend
from plot_styles.core.style_axis import save_axis
from plot_styles.style import MODEL_COLORS, HEAD_LINESTYLES


BITMAP_FORMATS = {"png", "jpg", "jpeg", "tiff", "bmp"}


def _with_ext(name: str, file_format: str) -> str:
    root, ext = os.path.splitext(name)
    return name if ext else f"{root}.{file_format}"


def _save_kwargs(file_format: str, dpi: int | None):
    if file_format.lower() in BITMAP_FORMATS and dpi is not None:
        return {"dpi": dpi}
    return {}


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
):
    """Render a standalone legend shared by a task's plots."""

    fig = plot_only_legend(
        fig_size=(7.5, 2.5),
        active_models=model_order,
        train_sizes=train_sizes,
        dataset_markers=dataset_markers,
        dataset_pretty=dataset_pretty,
        model_colors=MODEL_COLORS,
        head_order=head_order,
        head_linestyles=HEAD_LINESTYLES,
        legends=legends,
    )

    os.makedirs(plot_dir, exist_ok=True)
    legend_path = os.path.join(plot_dir, _with_ext(f_name, file_format))
    fig.savefig(legend_path, bbox_inches="tight", **_save_kwargs(file_format, dpi))
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
):
    from plot_styles.style import QE_DATASET_MARKERS, QE_DATASET_PRETTY

    QE_TRAIN_SIZE = [15, 148, 1475, 2950]  # in thousands
    # QE_TRAIN_SIZE = [15, 148, 1475]  # in thousands

    QE_MODEL_ORDER = ["Nominal", "Scratch", "SSL", "Ref."]

    plot_dir = os.path.join(output_root, "QE")

    fig, axes, active_models = plot_loss(
        data,
        train_sizes=QE_TRAIN_SIZE,
        model_order=QE_MODEL_ORDER,
        dataset_markers=QE_DATASET_MARKERS,
        dataset_pretty=QE_DATASET_PRETTY,
        y_min=0.82,
        fig_size=(7, 6),
        grid=True,
    )

    if with_legend:
        plot_legend(
            fig,
            active_models=active_models,
            train_sizes=QE_TRAIN_SIZE,
            dataset_markers=QE_DATASET_MARKERS,
            dataset_pretty=QE_DATASET_PRETTY,
            model_colors=MODEL_COLORS,
            legends=["dataset", "models"],
        )
    os.makedirs(plot_dir, exist_ok=True)
    fig.savefig(os.path.join(plot_dir, _with_ext("loss", file_format)), bbox_inches="tight", **_save_kwargs(file_format, dpi))
    if save_individual_axes:
        for idx, axis in enumerate(axes):
            save_axis(axis, plot_dir, f_name=_with_ext(f"loss_line_panel_{idx + 1}", file_format), dpi=dpi)

    fig_pair, axes_pair, active_pair = plot_bar_line(
        data_df=data,
        metric="pairing",
        model_order=QE_MODEL_ORDER,
        train_sizes=QE_TRAIN_SIZE,
        dataset_markers=QE_DATASET_MARKERS,
        dataset_pretty=QE_DATASET_PRETTY,
        y_label="Pairing Efficiency [%]",
        x_label="Train Size [K]",
        y_min=(78.0, 20.0),
        logx=True,
        panel_ratio=(2, 2),  # << tunable left:right panel ratio
        x_indicator=1e3,  # << typical dataset size indicator
    )
    if with_legend:
        plot_legend(
            fig_pair,
            active_models=active_pair,
            train_sizes=QE_TRAIN_SIZE,
            dataset_markers=QE_DATASET_MARKERS,
            dataset_pretty=QE_DATASET_PRETTY,
            model_colors=MODEL_COLORS,
            legends=["dataset", "models"],
        )
    fig_pair.savefig(os.path.join(plot_dir, _with_ext("pair", file_format)), bbox_inches="tight", **_save_kwargs(file_format, dpi))
    if save_individual_axes:
        for idx, axis in enumerate(axes_pair):
            save_axis(axis, plot_dir, f_name=_with_ext(f"pair_panel_{idx + 1}", file_format), dpi=dpi)

    fig_delta, axes_delta, active_delta = plot_bar_line(
        data_df=data,
        metric="deltaD",
        model_order=QE_MODEL_ORDER,
        train_sizes=QE_TRAIN_SIZE,
        dataset_markers=QE_DATASET_MARKERS,
        dataset_pretty=QE_DATASET_PRETTY,
        y_label=r"precision on D [%]",
        x_label="Train Size [K]",
        y_min=(1.0, 1.0),
        panel_ratio=(2, 2),  # << tunable left:right panel ratio
        x_indicator=1e3,  # << typical dataset size indicator
        logy=False,
        logx=True,
    )
    if with_legend:
        plot_legend(
            fig_delta,
            active_models=active_delta,
            train_sizes=QE_TRAIN_SIZE,
            dataset_markers=QE_DATASET_MARKERS,
            dataset_pretty=QE_DATASET_PRETTY,
            model_colors=MODEL_COLORS,
            legends=["dataset", "models"],
        )
    fig_delta.savefig(os.path.join(plot_dir, _with_ext("deltaD", file_format)), bbox_inches="tight", **_save_kwargs(file_format, dpi))
    if save_individual_axes:
        for idx, axis in enumerate(axes_delta):
            save_axis(axis, plot_dir, f_name=_with_ext(f"deltaD_panel_{idx + 1}", file_format), dpi=dpi)

    return {
        "loss": {"fig": fig, "axes": axes, "active_models": active_models},
        "pair": {"fig": fig_pair, "axes": axes_pair, "active_models": active_pair},
        "delta": {"fig": fig_delta, "axes": axes_delta, "active_models": active_delta},
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
):
    from plot_styles.style import BSM_DATASET_MARKERS, BSM_DATASET_PRETTY

    # BSM_TRAIN_SIZE = [10, 30, 100, 300, 1000]  # in thousands
    BSM_TRAIN_SIZE = [10, 30, 100, 300]  # in thousands
    BSM_TYPICAL_DATASET_SIZE = 100  # in thousands

    # BSM_MODEL = ["Nominal", "Ablation", "SSL", "Scratch", "SPANet"]
    BSM_MODEL = ["Nominal", "Scratch", "SSL", "SPANet"]
    # BSM_MODEL = ["Nominal", "Ablation"]
    BSM_HEAD = ["Cls", "Cls+Asn"]
    # BSM_HEAD = ["Cls", "Cls+Asn", "Cls+Seg", "Cls+Asn+Seg"]

    plot_dir = os.path.join(output_root, "BSM")

    fig, axes, active_models = plot_loss(
        data[data["mass_a"] == "30"],
        train_sizes=BSM_TRAIN_SIZE,
        model_order=BSM_MODEL,
        dataset_markers=BSM_DATASET_MARKERS,
        dataset_pretty=BSM_DATASET_PRETTY,
        fig_size=(12, 6),
        multi_panel_config={
            "n_rows": 1,
            "n_cols": 2,
            "configs": [
                *BSM_HEAD
                # "Cls+Seg", "Cls+Assign+Seg"
            ]
        },
        grid=True,
    )
    if with_legend:
        plot_legend(
            fig,
            active_models=active_models,
            train_sizes=BSM_TRAIN_SIZE,
            dataset_markers=BSM_DATASET_MARKERS,
            dataset_pretty=BSM_DATASET_PRETTY,
            model_colors=MODEL_COLORS,
            head_order=BSM_HEAD,
            head_linestyles=HEAD_LINESTYLES,
            legends=["dataset", "heads", "models"],
        )
    os.makedirs(plot_dir, exist_ok=True)
    fig.savefig(os.path.join(plot_dir, _with_ext("loss", file_format)), bbox_inches="tight", **_save_kwargs(file_format, dpi))
    if save_individual_axes:
        for idx, axis in enumerate(axes):
            save_axis(axis, plot_dir, f_name=_with_ext(f"loss_line_panel_{idx + 1}", file_format), dpi=dpi)

    fig_pair, axes_pair, active_pair = plot_bar_line(
        data_df=data[data["mass_a"] == "30"],
        metric="pairing",
        model_order=BSM_MODEL,
        train_sizes=BSM_TRAIN_SIZE,
        dataset_markers=BSM_DATASET_MARKERS,
        dataset_pretty=BSM_DATASET_PRETTY,
        # head_order=["Cls+Asn", "Cls+Asn+Seg"],
        head_order=["Cls+Asn"],
        y_label="Pairing Efficiency [%]",
        y_min=(65, 20),
        logx=True,
        panel_ratio=(2, 2),  # << tunable left:right panel ratio
        x_indicator=BSM_TYPICAL_DATASET_SIZE,  # << typical dataset size indicator
    )
    if with_legend:
        plot_legend(
            fig_pair,
            active_models=active_pair,
            train_sizes=BSM_TRAIN_SIZE,
            dataset_markers=BSM_DATASET_MARKERS,
            dataset_pretty=BSM_DATASET_PRETTY,
            model_colors=MODEL_COLORS,
            head_order=["Cls+Asn"],
            head_linestyles=HEAD_LINESTYLES,
            legends=["dataset", "heads", "models"],
        )
    fig_pair.savefig(os.path.join(plot_dir, _with_ext("pair", file_format)), bbox_inches="tight", **_save_kwargs(file_format, dpi))
    if save_individual_axes:
        for idx, axis in enumerate(axes_pair):
            save_axis(axis, plot_dir, f_name=_with_ext(f"pair_panel_{idx + 1}", file_format), dpi=dpi)

    fig_sic, axes_sic, active_sic = sic_plot(
        data[data["mass_a"] == "30"],
        model_order=BSM_MODEL,
        train_sizes=BSM_TRAIN_SIZE,
        head_order=BSM_HEAD,
        dataset_markers=BSM_DATASET_MARKERS,
        dataset_pretty=BSM_DATASET_PRETTY,
        x_indicator=BSM_TYPICAL_DATASET_SIZE,
        y_min=[0, 0, 0.85],
        fig_size=(21, 6),
        plot_dir="plot/BSM",
        f_name=_with_ext("sic", file_format),
        with_legend=False,
        save_individual_axes=save_individual_axes,
        file_format=file_format,
        dpi=dpi,
    )
    if with_legend:
        plot_legend(
            fig_sic,
            active_models=active_sic,
            train_sizes=BSM_TRAIN_SIZE,
            dataset_markers=BSM_DATASET_MARKERS,
            dataset_pretty=BSM_DATASET_PRETTY,
            model_colors=MODEL_COLORS,
            head_order=BSM_HEAD,
            head_linestyles=HEAD_LINESTYLES,
            legends=["dataset", "heads", "models"],
        )
    fig_sic.savefig(os.path.join(plot_dir, _with_ext("sic", file_format)), bbox_inches="tight", **_save_kwargs(file_format, dpi))
    if save_individual_axes:
        for axis_name, axis in zip(["sic_curve", "sic_bar", "sic_scatter"], axes_sic):
            save_axis(axis, plot_dir, f_name=_with_ext(axis_name, file_format), dpi=dpi)

    return {
        "loss": {"fig": fig, "axes": axes, "active_models": active_models},
        "pair": {"fig": fig_pair, "axes": axes_pair, "active_models": active_pair},
        "sic": {"fig": fig_sic, "axes": axes_sic, "active_models": active_sic},
    }


def plot_qe_results_webpage(
    data,
    *,
    output_root: str = "plot",
    file_format: str = "pdf",
    dpi: int | None = None,
):
    from plot_styles.style import QE_DATASET_MARKERS, QE_DATASET_PRETTY

    legend_cfg = {
        "models": ["Nominal", "Scratch", "SSL", "Ref."],
        "heads": ["Cls", "Cls+Asn"],
        "train_sizes": [15, 148, 1475, 2950],
    }

    plot_dir = os.path.join(output_root, "QE")

    legend_path = plot_task_legend(
        plot_dir=plot_dir,
        model_order=legend_cfg["models"],
        train_sizes=legend_cfg["train_sizes"],
        dataset_markers=QE_DATASET_MARKERS,
        dataset_pretty=QE_DATASET_PRETTY,
        head_order=legend_cfg["heads"],
        legends=["dataset", "heads", "models"],
        file_format=file_format,
        dpi=dpi,
    )

    results = plot_qe_results(
        data,
        output_root=output_root,
        file_format=file_format,
        dpi=dpi,
        save_individual_axes=True,
        with_legend=False,
    )

    return {
        "legend": legend_path,
        "loss": [os.path.join(plot_dir, _with_ext(f"loss_line_panel_{idx + 1}", file_format)) for idx, _ in enumerate(results["loss"]["axes"])],
        "pair": [os.path.join(plot_dir, _with_ext(f"pair_panel_{idx + 1}", file_format)) for idx, _ in enumerate(results["pair"]["axes"])],
        "delta": [os.path.join(plot_dir, _with_ext(f"deltaD_panel_{idx + 1}", file_format)) for idx, _ in enumerate(results["delta"]["axes"])],
    }


def plot_bsm_results_webpage(
    data,
    *,
    output_root: str = "plot",
    file_format: str = "pdf",
    dpi: int | None = None,
):
    from plot_styles.style import BSM_DATASET_MARKERS, BSM_DATASET_PRETTY

    legend_cfg = {
        "models": ["Nominal", "Scratch", "SSL", "Ref."],
        "heads": ["Cls", "Cls+Asn"],
        "train_sizes": [10, 30, 100, 300],
    }

    plot_dir = os.path.join(output_root, "BSM")

    legend_path = plot_task_legend(
        plot_dir=plot_dir,
        model_order=legend_cfg["models"],
        train_sizes=legend_cfg["train_sizes"],
        dataset_markers=BSM_DATASET_MARKERS,
        dataset_pretty=BSM_DATASET_PRETTY,
        head_order=legend_cfg["heads"],
        legends=["dataset", "heads", "models"],
        file_format=file_format,
        dpi=dpi,
    )

    results = plot_bsm_results(
        data,
        output_root=output_root,
        file_format=file_format,
        dpi=dpi,
        save_individual_axes=True,
        with_legend=False,
    )

    return {
        "legend": legend_path,
        "loss": [os.path.join(plot_dir, _with_ext(f"loss_line_panel_{idx + 1}", file_format)) for idx, _ in enumerate(results["loss"]["axes"])],
        "pair": [os.path.join(plot_dir, _with_ext(f"pair_panel_{idx + 1}", file_format)) for idx, _ in enumerate(results["pair"]["axes"])],
        "sic": [
            os.path.join(plot_dir, _with_ext(name, file_format))
            for name in ["sic_curve", "sic_bar", "sic_scatter"]
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
):
    AD_MODEL = ["Nominal", "Scratch", "SSL"]
    # AD_MODEL = ["Nominal", "Scratch", "SSL", "Ablation"]

    plot_dir = os.path.join(output_root, "AD")

    plot_ad_sig_summary(
        data['sig'],
        models_order=AD_MODEL,
        channels_order=["train-OS-test-OS", "train-SS-test-OS", "train-OS-test-SS", "train-SS-test-SS"],
        show_error=True, var="median", f_name=_with_ext("ad_significance", file_format), plot_dir=plot_dir,
        dpi=dpi,
        file_format=file_format,
        y_ref=6.4
    )

    plot_ad_gen_summary(
        data['gen'],
        models_order=AD_MODEL,
        label_right="calibration magnitude [%]",
        f_name=_with_ext("ad_generation", file_format),
        plot_dir=plot_dir,
        y_min_left=0.6,
        y_min_right=0,
        save_individual_axes=save_individual_axes,
        dpi=dpi,
        file_format=file_format,
    )

    pass


def plot_ad_results_webpage(
    data,
    *,
    output_root: str = "plot",
    file_format: str = "pdf",
    dpi: int | None = None,
):
    plot_dir = os.path.join(output_root, "AD")

    plot_ad_results(
        data,
        output_root=output_root,
        file_format=file_format,
        dpi=dpi,
        save_individual_axes=True,
    )

    return {
        "legend": None,
        "sig": [os.path.join(plot_dir, _with_ext("ad_significance", file_format))],
        "gen": [os.path.join(plot_dir, _with_ext("ad_generation", file_format))],
    }


if __name__ == '__main__':
    # qe_data = read_qe_data('data/QE_results_table.csv')
    # plot_qe_results(qe_data)

    bsm_data = read_bsm_data('data/BSM')
    plot_bsm_results(bsm_data)

    # ad_data = read_ad_data("data/AD")
    # plot_ad_results(ad_data)

    pass
