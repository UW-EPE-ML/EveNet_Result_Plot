import re
import json
import os
import pandas as pd
import numpy as np
from functools import reduce

from plot_styles.loss_line import plot_loss
from plot_styles.bar_line_two_panel import plot_bar_line
from plot_styles.sic import sic_plot
from plot_styles.ad_bar import plot_ad_sig_summary, plot_ad_gen_summary


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


def plot_qe_results(data):
    from plot_styles.style import QE_DATASET_MARKERS, QE_DATASET_PRETTY

    QE_TRAIN_SIZE = [43, 130, 302, 432]  # in thousands

    plot_loss(
        data,
        train_sizes=QE_TRAIN_SIZE,
        model_order=["Nominal", "Scratch"],
        dataset_markers=QE_DATASET_MARKERS,
        dataset_pretty=QE_DATASET_PRETTY,
        y_min=0.82, fig_size=(7, 6), plot_dir="plot/QE", f_name="loss.pdf",
        grid=True,
        with_legend=True,
        save_individual_axes=True,
    )

    plot_bar_line(
        data_df=data,
        metric="pairing",
        model_order=["Nominal", "Scratch", "Ref."],
        train_sizes=QE_TRAIN_SIZE,
        dataset_markers=QE_DATASET_MARKERS,
        dataset_pretty=QE_DATASET_PRETTY,
        y_label="Pairing Efficiency [%]",
        y_min=42,
        panel_ratio=(2, 2),  # << tunable left:right panel ratio
        x_indicator=2.5e2,  # << typical dataset size indicator
        plot_dir="plot/QE",
        f_name="pair.pdf",
        save_individual_axes=True,
    )

    plot_bar_line(
        data_df=data,
        metric="deltaD",
        model_order=["Nominal", "Scratch", "Ref."],
        train_sizes=QE_TRAIN_SIZE,
        dataset_markers=QE_DATASET_MARKERS,
        dataset_pretty=QE_DATASET_PRETTY,
        y_label=r"precision on D [%]",
        y_min=.5,
        panel_ratio=(2, 2),  # << tunable left:right panel ratio
        x_indicator=2.5e2,  # << typical dataset size indicator
        logy=False,
        plot_dir="plot/QE",
        f_name="deltaD.pdf",
        save_individual_axes=True,
    )


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
                row["effective_step"] = epoch_data.get(
                    f'evenet-ma{row["mass_a"]}-{row["raw_model"].replace("evenet-", "")}-{row["raw_dataset_size"]}-{'Assign' in row['head']}-{'Seg' in row['head']}'
                    , {}).get("effective_steps", None)
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


def plot_bsm_results(data):
    from plot_styles.style import BSM_DATASET_MARKERS, BSM_DATASET_PRETTY

    # BSM_TRAIN_SIZE = [10, 30, 100, 300, 1000]  # in thousands
    BSM_TRAIN_SIZE = [30, 100, 300, 1000]  # in thousands
    BSM_TYPICAL_DATASET_SIZE = 250  # in thousands

    BSM_MODEL = ["Nominal", "Scratch", "SPANet"]
    BSM_HEAD = ["Cls", "Cls+Asn"]

    plot_loss(
        data[data["mass_a"] == "30"],
        train_sizes=BSM_TRAIN_SIZE,
        model_order=BSM_MODEL,
        dataset_markers=BSM_DATASET_MARKERS,
        dataset_pretty=BSM_DATASET_PRETTY,
        fig_size=(12, 6), plot_dir="plot/BSM", f_name="loss.pdf",
        multi_panel_config={
            "n_rows": 1,
            "n_cols": 2,
            "configs": [
                *BSM_HEAD
                # "Cls+Seg", "Cls+Assign+Seg"
            ]
        },
        grid=True,
        with_legend=True,
        save_individual_axes=True,
    )

    plot_bar_line(
        data_df=data[data["mass_a"] == "30"],
        metric="pairing",
        model_order=BSM_MODEL,
        train_sizes=BSM_TRAIN_SIZE,
        dataset_markers=BSM_DATASET_MARKERS,
        dataset_pretty=BSM_DATASET_PRETTY,
        head_order=["Cls+Asn", "Cls+Asn+Seg"],
        y_label="Pairing Efficiency [%]",
        y_min=(65, 50),
        logx=True,
        panel_ratio=(2, 2),  # << tunable left:right panel ratio
        x_indicator=BSM_TYPICAL_DATASET_SIZE,  # << typical dataset size indicator
        plot_dir="plot/BSM",
        f_name="pair.pdf",
        save_individual_axes=True,
    )

    sic_plot(
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
        f_name="sic.pdf",
        with_legend=True,
        save_individual_axes=True,
    )


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


def plot_ad_results(data):
    AD_MODEL = ["Nominal", "Scratch"]

    plot_ad_sig_summary(
        data['sig'],
        models_order=AD_MODEL,
        channels_order=["train-OS-test-OS", "train-SS-test-OS", "train-OS-test-SS", "train-SS-test-SS"],
        show_error=True, var="median", f_name="ad_significance.pdf", plot_dir="plot/AD",
        y_ref=6.4
    )

    plot_ad_gen_summary(
        data['gen'],
        models_order=AD_MODEL,
        label_right="calibration magnitude [%]",
        f_name="ad_generation.pdf",
        plot_dir="plot/AD",
        y_min_left=0.6,
        y_min_right=0,
        save_individual_axes=True,
    )

    pass


if __name__ == '__main__':
    qe_data = read_qe_data('data/QE_results_table.csv')
    plot_qe_results(qe_data)

    bsm_data = read_bsm_data('data/BSM')
    plot_bsm_results(bsm_data)

    ad_data = read_ad_data("data/AD")
    plot_ad_results(ad_data)

    pass
