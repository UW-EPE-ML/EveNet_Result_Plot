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
    "SSL":      "#5B84B1",   # muted steel blue
    "Nominal":  "#7CA2C3",   # softer blue-grey, same hue family
    "Scratch":  "#D7A46F",   # soft clay orange (muted)
    "Ablation": "#C28E6A",   # slightly darker orange, same hue family
    "Ref.":     "#9A9A9A",   # neutral grey
    "SPANet":   "#9A9A9A",   # same grey for grouping
}

MODEL_PRETTY = {
    "SSL": "SSL",
    "Nominal": "Nominal",
    "Ablation": "Ablation",
    "Scratch": "Scratch",
    "Ref.": "Ref",
    "SPANet": "SPANet",
}

QE_DATASET_MARKERS = {
    '43': "o",
    '130': "s",
    '302': "D",
    '432': "^",
}

QE_DATASET_PRETTY = {
    '43': "1%",
    '130': "50%",
    '302': "100%",
    '432': "200%",
}

BSM_DATASET_MARKERS = {
    '10': "o",
    '30': "P",
    '100': "s",
    '300': "D",
    '1000': "^",
}

BSM_DATASET_PRETTY = {
    '10': "1%",
    '30': "3%",
    '100': "10%",
    '300': "30%",
    '1000': "100%",
}

HEAD_LINESTYLES = {
    "Cls": "-",
    "Cls+Seg": "-.",
    "Cls+Assign": "--",
    "Cls+Assign+Seg": ":",
}