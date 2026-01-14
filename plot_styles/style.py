# MODEL_COLORS = {
#     "SSL": "#7399f4",
#     "Nominal": "#925eb0",
#     "Scratch": "#7ab656",
#     "Ablation": "#cc7c71",
#     "Ref.": "#a5aeb7",
#     "SPANet": "#a5aeb7",
# }

# MODEL_COLORS = {
#     "SSL":     "#784A71",   # muted blue
#     "Nominal": "#599AB9",   # muted orange
#     "Scratch": "#CA9B22",   # soft green
#     "Ablation": "#FF7F0E",  # muted red
#     "Ref.":    "#817F85",   # neutral grey
#     "SPANet":  "#817F85",   # same as Ref.
# }

BASE_MODEL_COLORS = {
    "SSL":      "#EDA4A1",   # distinct muted teal-green (large separation)
    "Nominal":  "#5F8FD9",   # fixed
    "Ablation": "#3E5F78",   # darker blue-grey for strong contrast
    "Scratch":  "#E3C565",   # fixed
    "Ref.":     "#9A9A9A",   # neutral grey
    "SPANet":   "#9A9A9A",   # same grey for grouping
    "XGBoost":  "#A9D3AD",
    "TabPFN":   "#9A9A9A",
}

MODEL_COLORS = dict(BASE_MODEL_COLORS)
MODEL_COLORS.update({
    "evenet-scratch_individual": BASE_MODEL_COLORS["Scratch"],
    "evenet-pretrain_individual": BASE_MODEL_COLORS["Nominal"],
    "evenet-pretrain_param": BASE_MODEL_COLORS.get("SSL", BASE_MODEL_COLORS["Nominal"]),
    "evenet-scratch_param": BASE_MODEL_COLORS.get("Scratch", BASE_MODEL_COLORS["Nominal"]),
    "xgb_individual": BASE_MODEL_COLORS["XGBoost"],
    "xgb_param": BASE_MODEL_COLORS["XGBoost"],
    "tabpfn_individual": BASE_MODEL_COLORS["TabPFN"],
})

MODEL_PRETTY = {
    "SSL": "SSL",
    "Nominal": "Full",
    "Ablation": "Ablation",
    "Scratch": "Scratch",
    "Ref.": "Ref",
    "SPANet": "SPANet",
    "XGBoost": "XGBoost",
    "TabPFN": "TabPFN",
    "evenet-full": "Nominal",
    "evenet-scratch_individual": "Scratch",
    "evenet-pretrain_individual": "Full",
    "evenet-pretrain_param": "Full (param)",
    "evenet-scratch_param": "Scratch (param)",
    "xgb_individual": "XGBoost",
    "xgb_param": "XGBoost (param)",
    "tabpfn_individual": "TabPFN",
}

QE_DATASET_MARKERS = {
    '15': "o",
    '148': "s",
    '1475': "D",
    '2950': "^",
}

QE_DATASET_PRETTY = {
    '15': "1%",
    '148': "15%",
    '1475': "150%",
    '2950': "300%",
}

BSM_DATASET_MARKERS = {
    '10': "o",
    '30': "P",
    '100': "s",
    '300': "D",
    '1000': "^",
}

BSM_DATASET_PRETTY = {
    '10': "5%",
    '30': "15%",
    '100': "50%",
    '300': "150%",
    '1000': "500%",
}

HEAD_LINESTYLES = {
    "Cls": "-",
    "Cls+Seg": "-.",
    "Cls+Asn": "--",
    "Cls+Asn+Seg": ":",
}
