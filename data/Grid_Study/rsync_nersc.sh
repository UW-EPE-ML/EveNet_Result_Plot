#!/bin/bash

base_dir="/pscratch/sd/t/tihsu/database/Grid_Study_CMS_OpenData_bbWW_HWW/method_arxiv"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rsync -Pavz \
  --include='*/' \
  --include='eval_metrics_*.json' \
  --exclude='*' \
  nersc:"$base_dir" "$script_dir"


rsync -Pavz "nersc:/pscratch/sd/t/tihsu/database/Grid_Study_CMS_OpenData_bbWW_HWW/data/cutflow.json" "$script_dir"