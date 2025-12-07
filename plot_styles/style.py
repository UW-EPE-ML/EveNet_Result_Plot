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

MODEL_COLORS = {
    "SSL":      "#EDA4A1",   # distinct muted teal-green (large separation)
    "Nominal":  "#5F8FD9",   # fixed
    "Ablation": "#3E5F78",   # darker blue-grey for strong contrast
    "Scratch":  "#E3C565",   # fixed
    "Ref.":     "#9A9A9A",   # neutral grey
    "SPANet":   "#9A9A9A",   # same grey for grouping
}

MODEL_PRETTY = {
    "SSL": "SSL",
    "Nominal": "Full",
    "Ablation": "Ablation",
    "Scratch": "Scratch",
    "Ref.": "Ref",
    "SPANet": "SPANet",
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
    '10': "4%",
    '30': "12%",
    '100': "40%",
    '300': "120%",
    '1000': "400%",
}

HEAD_LINESTYLES = {
    "Cls": "-",
    "Cls+Seg": "-.",
    "Cls+Asn": "--",
    "Cls+Asn+Seg": ":",
}