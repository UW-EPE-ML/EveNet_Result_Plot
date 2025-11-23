import re
import json
import os
import pandas as pd
import numpy as np

from plot_styles.loss_line import plot_loss
from plot_styles.bar_line_two_panel import plot_bar_line
from plot_styles.sic import sic_plot


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
        y_min=0.82, fig_size=(9, 8), plot_dir="plot/QE", f_name="loss.pdf",
        grid=True,
        with_legend=True,
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
        bar_margin=10.0,  # << bars closer to boundary
        bar_width=5.0,  # << slimmer bars
        bar_spacing=5.5,
        x_range=[2.5, 1.5],  # << small gap between bars,
        x_indicator=2.5e2,  # << typical dataset size indicator
        plot_dir="plot/QE",
        f_name="pair.pdf"
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
        bar_margin=10.0,  # << bars closer to boundary
        bar_width=5.0,  # << slimmer bars
        bar_spacing=5.5,
        x_range=[2.5, 2.5],  # << small gap between bars
        x_indicator=2.5e2,  # << typical dataset size indicator
        logy=False,
        plot_dir="plot/QE",
        f_name="deltaD.pdf"
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
        # {"name": "Cls", "use_assignment": False, "use_segmentation": False},
        # {"name": "Cls + Seg", "use_assignment": False, "use_segmentation": True},
        # {"name": "Cls + Assg", "use_assignment": True, "use_segmentation": False},
        # {"name": "Cls + Seg + Assg", "use_assignment": True, "use_segmentation": True},
        (False, False): "Cls",
        (False, True): "Cls+Seg",
        (True, False): "Cls+Assign",
        (True, True): "Cls+Assign+Seg",
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
            # "assignment": params["assignment"] == "on",
            # "segmentation": params["segmentation"] == "on",
            "head": HEAD_NAME_MAPPING[(params["assignment"] == "on", params["segmentation"] == "on")],
            "raw_dataset_size": params["size"],
            "train_size": int(float(params["size"]) * total_N),
        }

        return row

    dummy_epoch = 50
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
                    # "epoch": dummy_epoch,
                    # "effective_step": epoch_data.get(folder, dummy_epoch),
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

    # Merge loss and sic dataframes according to model, assignment, segmentation, train_size, mass_a
    df = pd.merge(
        df_loss,
        df_sic,
        on=["model", "head", "train_size", "mass_a"],
        how="outer"
    )

    # df['effective_step'] = df.apply(
    #     lambda row: convert_epochs_to_steps(
    #         row['epoch'],
    #         row['train_size'],  # full dataset size is 432k
    #         batch_size_per_GPU=2048,
    #         GPUs=4,
    #     ),
    #     axis=1
    # )

    return df


def plot_bsm_results(data):
    from plot_styles.style import BSM_DATASET_MARKERS, BSM_DATASET_PRETTY

    # BSM_TRAIN_SIZE = [10, 30, 100, 300, 1000]  # in thousands
    BSM_TRAIN_SIZE = [30, 100, 300, 1000]  # in thousands

    BSM_MODEL = ["Nominal", "Scratch", "SPANet"]
    BSM_HEAD = ["Cls", "Cls+Assign"]

    plot_loss(
        data[data["mass_a"] == "30"],
        train_sizes=BSM_TRAIN_SIZE,
        model_order=BSM_MODEL,
        dataset_markers=BSM_DATASET_MARKERS,
        dataset_pretty=BSM_DATASET_PRETTY,
        fig_size=(11, 5), plot_dir="plot/BSM", f_name="loss.pdf",
        multi_panel_config={
            "n_rows": 1,
            "n_cols": 2,
            "configs": [
                "Cls", "Cls+Assign",
                # "Cls+Seg", "Cls+Assign+Seg"
            ]
        },
        grid=True,
        with_legend=True,
    )

    sic_plot(
        data[data["mass_a"] == "30"],
        model_order=BSM_MODEL,
        train_sizes=BSM_TRAIN_SIZE,
        head_order=BSM_HEAD,
        dataset_markers=BSM_DATASET_MARKERS,
        dataset_pretty=BSM_DATASET_PRETTY,
        fig_size=(21, 6),
        plot_dir="plot/BSM",
        f_name="sic.pdf",
        with_legend=True,
    )


if __name__ == '__main__':
    qe_data = read_qe_data('data/QE_results_table.csv')
    plot_qe_results(qe_data)

    bsm_data = read_bsm_data('data/BSM')
    plot_bsm_results(bsm_data)
