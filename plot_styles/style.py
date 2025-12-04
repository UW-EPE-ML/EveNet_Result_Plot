# MODEL_COLORS = {
#     "SSL": "#7399f4",
#     "Nominal": "#925eb0",
#     "Scratch": "#7ab656",
#     "Ablation": "#cc7c71",
#     "Ref.": "#a5aeb7",
#     "SPANet": "#a5aeb7",
# }

# MODEL_COLORS = {
#     "SSL":     "#4C72B0",   # muted blue
#     "Nominal": "#DD8452",   # muted orange
#     "Scratch": "#55A868",   # soft green
#     "Ablation": "#C44E52",  # muted red
#     "Ref.":    "#817F85",   # neutral grey
#     "SPANet":  "#817F85",   # same as Ref.
# }

MODEL_COLORS = {
    "SSL":      "#6BAF8C",   # distinct muted teal-green (large separation)
    "Nominal":  "#7CA2C3",   # fixed
    "Ablation": "#3E5F78",   # darker blue-grey for strong contrast
    "Scratch":  "#D7A46F",   # fixed
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
    '148': "50%",
    '1475': "100%",
    '2950': "200%",
}

BSM_DATASET_MARKERS = {
    '10': "o",
    '30': "P",
    '100': "s",
    '300': "D",
    '1000': "^",
}

BSM_DATASET_PRETTY = {
    '10': "10%",
    '30': "30%",
    '100': "100%",
    '300': "300%",
    '1000': "1000%",
}

HEAD_LINESTYLES = {
    "Cls": "-",
    "Cls+Seg": "-.",
    "Cls+Asn": "--",
    "Cls+Asn+Seg": ":",
}