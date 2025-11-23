MODEL_COLORS = {
    "SSL": "#7399f4",
    "Nominal": "#925eb0",
    "Scratch": "#7ab656",
    "Ablation": "#cc7c71",
    "Ref.": "#a5aeb7",
    "SPANet": "#a5aeb7",
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