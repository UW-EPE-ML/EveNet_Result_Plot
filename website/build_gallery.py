"""Build a static plot gallery for GitHub Pages.

This script regenerates PNG renderings of the paper plots from the
repository data and assembles a simple static HTML gallery. It is used by
CI to publish the figures to GitHub Pages, but can also be run locally
for spot-checking:

    python website/build_gallery.py --output-dir site --format png --dpi 200
"""
from __future__ import annotations

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

# Ensure imports work whether the script is run from the repository root or
# directly inside the website/ folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

# Use a headless backend for CI and local builds without a display
matplotlib.use("Agg")

from paper_plot import (  # noqa: E402  # pylint: disable=wrong-import-position
    DEFAULT_AD_CONFIG,
    DEFAULT_BSM_CONFIG,
    DEFAULT_QE_CONFIG,
    read_ad_data,
    read_bsm_data,
    read_qe_data,
    plot_ad_results_webpage,
    plot_bsm_results_webpage,
    plot_qe_results_webpage,
)
from plot_styles.sic import compute_sic_with_unc  # noqa: E402  # pylint: disable=wrong-import-position


def _render_index(output_dir: Path, sections: List[Dict[str, object]]) -> None:
    """Write a minimal HTML page to preview all generated plots."""

    cards = []
    for section in sections:
        card_items = "\n".join([_render_plot_card(plot) for plot in section["plots"]])
        cards.append(
            f"<section id=\"{section['id']}\">"
            f"<h2>{section['title']}</h2>"
            f"<p>{section['blurb']}</p>"
            f"<div class=\"grid\">{card_items}</div>"
            "</section>"
        )

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>EveNet Plot Gallery</title>
  <style>
    :root {{
      color-scheme: light dark;
      font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
      background: #0b1021;
      color: #e8eaed;
    }}
    body {{
      margin: 0;
      padding: 1.5rem;
      line-height: 1.5;
      max-width: 1100px;
      margin-left: auto;
      margin-right: auto;
    }}
    header {{
      margin-bottom: 1.5rem;
    }}
    h1 {{ margin: 0 0 0.5rem 0; }}
    p.lede {{ opacity: 0.85; margin: 0 0 1rem 0; }}
    section {{ margin-bottom: 2rem; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 1rem;
      align-items: stretch;
    }}
    figure {{
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 8px;
      padding: 0.75rem;
      margin: 0;
      box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      height: 100%;
    }}
    figure .content-row {{
      display: flex;
      gap: 0.75rem;
      align-items: flex-start;
      flex: 1 1 auto;
    }}
    figure .content-row .media {{
      flex: 1 1 55%;
      min-width: 0;
    }}
    figure .content-row .table-container {{
      flex: 1 1 45%;
      overflow: auto;
      max-height: 380px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 6px;
      padding: 0.35rem;
      background: rgba(255, 255, 255, 0.02);
    }}
    @media (max-width: 768px) {{
      figure .content-row {{
        flex-direction: column;
      }}
      figure .content-row .table-container {{
        width: 100%;
        max-height: none;
      }}
    }}
    table.data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      color: #e2e8f0;
    }}
    table.data-table th,
    table.data-table td {{
      padding: 0.25rem 0.35rem;
      border: 1px solid rgba(255, 255, 255, 0.08);
      text-align: left;
      white-space: nowrap;
    }}
    table.data-table th {{
      background: rgba(255, 255, 255, 0.06);
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    figure img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 4px;
      background: #0f172a;
      border: 1px solid rgba(255, 255, 255, 0.08);
    }}
    figcaption {{
      font-size: 0.95rem;
      margin-top: 0.35rem;
      color: #cbd5e1;
    }}
    a {{ color: #7dd3fc; }}
  </style>
</head>
<body>
  <header>
    <h1>EveNet Plot Gallery</h1>
    <p class=\"lede\">Automatically rendered from the repository data. Each panel below links to the latest plots produced during the GitHub Pages build.</p>
  </header>
  {''.join(cards)}
</body>
</html>
"""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"Wrote gallery → {output_dir / 'index.html'}")


def _section(
    section_id: str,
    title: str,
    blurb: str,
    plots: List[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "id": section_id,
        "title": title,
        "blurb": blurb,
        "plots": plots,
    }


def _fmt(val):
    if isinstance(val, float):
        return f"{val:.4g}"
    return "" if val is None else str(val)


def _table_from_dicts(rows: List[Dict[str, object]], *, header_order: List[str] | None = None):
    if not rows:
        return None
    keys = header_order or list(dict.fromkeys(k for row in rows for k in row.keys()))
    formatted_rows = [[_fmt(row.get(k)) for k in keys] for row in rows]
    return {"headers": keys, "rows": formatted_rows}


def _render_table(table: Dict[str, object] | None) -> str:
    if not table:
        return ""
    header_html = "".join(f"<th>{h}</th>" for h in table["headers"])
    row_html = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in table["rows"]
    )
    return (
        "<div class=\"table-container\">"
        "<table class=\"data-table\">"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{row_html}</tbody>"
        "</table>"
        "</div>"
    )


def _render_plot_card(plot: Dict[str, object]) -> str:
    table_html = _render_table(plot.get("table"))
    caption = plot.get("caption", "")
    image_html = "" if plot.get("no_image") else (
        f"<div class=\"media\"><img src=\"{plot['src']}\" alt=\"{caption}\" loading=\"lazy\"></div>"
    )
    body_html = image_html
    if table_html:
        body_html = f"<div class=\"content-row\">{image_html}{table_html}</div>"
    return (
        "<figure>"
        f"{body_html}"
        f"<figcaption>{caption}</figcaption>"
        "</figure>"
    )


def _scatter_table(
    df,
    *,
    metric: str,
    train_sizes: List[int],
    model_order: List[str],
    head_order: List[str] | None = None,
) -> Dict[str, object] | None:
    head_iter = head_order or [None]
    rows = []
    unc_candidates = [f"{metric}_unc", "uncertainty"]
    unc_col = next((c for c in unc_candidates if c in df.columns), None)

    for model in model_order:
        df_model = df[df["model"] == model]
        if df_model.empty:
            continue
        for head in head_iter:
            df_head = df_model
            if head is not None and "head" in df_head.columns:
                df_head = df_head[df_head["head"] == head]
            df_head = df_head[df_head["train_size"].isin(train_sizes)]
            if df_head.empty:
                continue
            for _, row in df_head.sort_values("train_size").iterrows():
                rows.append(
                    {
                        "model": model,
                        **({"head": head} if head is not None else {}),
                        "train_size": row.get("train_size"),
                        metric: row.get(metric),
                        "error": row.get(unc_col) if unc_col else None,
                    }
                )
    headers = ["model"] + (["head"] if head_order else []) + ["train_size", metric, "error"]
    return _table_from_dicts(rows, header_order=headers)


def _sic_scatter_table(
    df,
    *,
    model_order: List[str],
    train_sizes: List[int],
    head_order: List[str] | None,
) -> Dict[str, object] | None:
    rows = []
    heads = head_order or [None]
    for model in model_order:
        df_model = df[df["model"] == model]
        if df_model.empty:
            continue
        for head in heads:
            df_head = df_model if head is None else df_model[df_model["head"] == head]
            for train_size in train_sizes:
                df_size = df_head[df_head["train_size"] == train_size]
                if df_size.empty:
                    continue
                row = df_size.iloc[0]
                TPR = np.asarray(row.get("TPR", []))
                FPR = np.asarray(row.get("FPR", []))
                if TPR.size == 0 or FPR.size == 0:
                    continue
                FPR_unc = np.asarray(row.get("FPR_unc", np.zeros_like(FPR)))
                sic, sic_unc = compute_sic_with_unc(TPR, FPR, FPR_unc)
                max_idx = int(np.argmax(sic))
                rows.append(
                    {
                        "model": model,
                        **({"head": head} if head_order else {}),
                        "train_size": train_size,
                        "sic": sic[max_idx],
                        "error": sic_unc[max_idx],
                    }
                )
    headers = ["model"] + (["head"] if head_order else []) + ["train_size", "sic", "error"]
    return _table_from_dicts(rows, header_order=headers)


def _sic_bar_table(
    df,
    *,
    model_order: List[str],
    train_sizes: List[int],
    head_order: List[str] | None,
    target_train_size: int,
) -> Dict[str, object] | None:
    rows = []
    heads = head_order or [None]
    for model in model_order:
        df_model = df[df["model"] == model]
        if df_model.empty:
            continue
        for head in heads:
            df_head = df_model if head is None else df_model[df_model["head"] == head]
            df_size = df_head[df_head["train_size"] == target_train_size]
            if df_size.empty:
                continue
            row = df_size.iloc[0]
            TPR = np.asarray(row.get("TPR", []))
            FPR = np.asarray(row.get("FPR", []))
            if TPR.size == 0 or FPR.size == 0:
                continue
            FPR_unc = np.asarray(row.get("FPR_unc", np.zeros_like(FPR)))
            sic, sic_unc = compute_sic_with_unc(TPR, FPR, FPR_unc)
            max_idx = int(np.argmax(sic))
            rows.append(
                {
                    "model": model,
                    **({"head": head} if head_order else {}),
                    "train_size": target_train_size,
                    "sic": sic[max_idx],
                    "error": sic_unc[max_idx],
                }
            )
    headers = ["model"] + (["head"] if head_order else []) + ["train_size", "sic", "error"]
    return _table_from_dicts(rows, header_order=headers)


def _bar_table(
    df,
    *,
    metric: str,
    model_order: List[str],
    head_order: List[str],
    train_size_for_bar: int,
) -> Dict[str, object] | None:
    rows = []
    for model in model_order:
        df_model = df[(df["model"] == model) & (df["train_size"] == train_size_for_bar)]
        if df_model.empty:
            continue
        for head in head_order:
            df_head = df_model
            if head is not None and "head" in df_head.columns:
                df_head = df_head[df_head["head"] == head]
            df_head = df_head.dropna(subset=[metric])
            if df_head.empty:
                continue
            df_head = df_head.sort_values(metric)
            y_val = df_head[metric].iloc[-1]
            rows.append(
                {
                    "model": model,
                    **({"head": head} if head is not None else {}),
                    "train_size": train_size_for_bar,
                    metric: y_val,
                }
            )
    headers = ["model"] + (["head"] if head_order and head_order != [None] else []) + ["train_size", metric]
    return _table_from_dicts(rows, header_order=headers)


def _loss_table(df, *, model_order: List[str]) -> Dict[str, object] | None:
    rows = []
    for model in model_order:
        df_model = df[df["model"] == model]
        if df_model.empty:
            continue
        head_values = [None]
        if "head" in df_model.columns:
            head_values = sorted(df_model["head"].dropna().unique())
        for head in head_values:
            df_head = df_model if head is None else df_model[df_model["head"] == head]
            for _, row in df_head.sort_values("effective_step").iterrows():
                rows.append(
                    {
                        "model": model,
                        **({"head": head} if head is not None else {}),
                        "effective_step": row.get("effective_step"),
                        "val_loss": row.get("val_loss"),
                    }
                )
    headers = ["model"] + (["head"] if any("head" in r for r in rows) else []) + ["effective_step", "val_loss"]
    return _table_from_dicts(rows, header_order=headers)


def _ad_significance_table(df) -> Dict[str, object] | None:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "model": row.get("model"),
                "channel": row.get("channel"),
                "calibrated": row.get("calibrated"),
                "median": row.get("median"),
                "lower_error": row.get("median") - row.get("lower") if not pd.isna(row.get("lower")) else None,
                "upper_error": row.get("upper") - row.get("median") if not pd.isna(row.get("upper")) else None,
            }
        )
    return _table_from_dicts(
        rows,
        header_order=["model", "channel", "calibrated", "median", "lower_error", "upper_error"],
    )


def _ad_generation_table(df, *, models_order: List[str]) -> Dict[str, object] | None:
    def agg(metric: str):
        result = []
        for model in models_order:
            for cal in [False, True]:
                for group in ["OS", "SS"]:
                    subset = df[(df["model"] == model) & (df["calibrated"] == cal) & (df["train_type"] == group)][
                        metric
                    ].dropna()
                    if subset.empty:
                        center = err_low = err_high = np.nan
                    else:
                        q16, q50, q84 = np.nanpercentile(subset, [16, 50, 84])
                        center, err_low, err_high = q50, q50 - q16, q84 - q50
                    result.append(
                        {
                            "metric": metric,
                            "model": model,
                            "calibrated": cal,
                            "group": group,
                            "value": center,
                            "lower_error": err_low,
                            "upper_error": err_high,
                        }
                    )
        return result

    rows = agg("mmd") + agg("mean_calibration_difference")
    return _table_from_dicts(
        rows,
        header_order=["metric", "model", "group", "calibrated", "value", "lower_error", "upper_error"],
    )


def main():
    parser = argparse.ArgumentParser(description="Generate static plot gallery")
    parser.add_argument("--output-dir", default="site", help="Directory to write the static site")
    parser.add_argument("--format", default="png", help="Image format for plot exports")
    parser.add_argument("--dpi", type=int, default=200, help="DPI to use for bitmap exports")
    parser.add_argument("--skip-qe", action="store_true", help="Skip QC/QE plots")
    parser.add_argument("--skip-bsm", action="store_true", help="Skip BSM plots")
    parser.add_argument("--skip-ad", action="store_true", help="Skip anomaly-detection plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    plots_root = output_dir / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    sections: List[Dict[str, object]] = []

    if not args.skip_qe and Path("data/QE_results_table.csv").is_file():
        print("Rendering QC/QE plots…")
        qe_data = read_qe_data("data/QE_results_table.csv")
        qe_outputs = plot_qe_results_webpage(
            qe_data,
            output_root=str(plots_root),
            file_format=args.format,
            dpi=args.dpi,
        )
        qe_loss_table = _loss_table(qe_data, model_order=DEFAULT_QE_CONFIG["models"])
        pair_bar_size = max(DEFAULT_QE_CONFIG["train_sizes"])
        qe_pair_tables = [
            _scatter_table(
                qe_data,
                metric=DEFAULT_QE_CONFIG["pair_scatter"]["metric"],
                train_sizes=DEFAULT_QE_CONFIG["train_sizes"],
                model_order=DEFAULT_QE_CONFIG["models"],
                head_order=DEFAULT_QE_CONFIG.get("heads") or None,
            ),
            _bar_table(
                qe_data,
                metric=DEFAULT_QE_CONFIG["pair_bar"]["metric"],
                model_order=DEFAULT_QE_CONFIG["models"],
                head_order=DEFAULT_QE_CONFIG.get("heads") or [None],
                train_size_for_bar=pair_bar_size,
            ),
        ]
        qe_delta_tables = [
            _scatter_table(
                qe_data,
                metric=DEFAULT_QE_CONFIG["delta_scatter"]["metric"],
                train_sizes=DEFAULT_QE_CONFIG["train_sizes"],
                model_order=DEFAULT_QE_CONFIG["models"],
                head_order=DEFAULT_QE_CONFIG.get("heads") or None,
            ),
            _bar_table(
                qe_data,
                metric=DEFAULT_QE_CONFIG["delta_bar"]["metric"],
                model_order=DEFAULT_QE_CONFIG["models"],
                head_order=DEFAULT_QE_CONFIG.get("heads") or [None],
                train_size_for_bar=pair_bar_size,
            ),
        ]
        qe_plots = [
            {"src": str(Path(qe_outputs["legend"]).relative_to(output_dir)), "caption": "QC/QE legend"},
            *[
                {
                    "src": str(Path(path).relative_to(output_dir)),
                    "caption": f"QC: loss panel {i + 1}",
                    "table": qe_loss_table,
                }
                for i, path in enumerate(qe_outputs["loss"])
            ],
            *[
                {
                    "src": str(Path(path).relative_to(output_dir)),
                    "caption": f"QC: pairing panel {i + 1}",
                    "table": qe_pair_tables[i],
                }
                for i, path in enumerate(qe_outputs["pair"])
            ],
            *[
                {
                    "src": str(Path(path).relative_to(output_dir)),
                    "caption": f"QC: DeltaD panel {i + 1}",
                    "table": qe_delta_tables[i],
                }
                for i, path in enumerate(qe_outputs["delta"])
            ],
        ]
        sections.append(
            _section(
                "qe",
                "Quantum-correlation benchmark (QC)",
                "Validation loss, pairing efficiency, and DeltaD precision across training sizes.",
                qe_plots,
            )
        )
    else:
        print("Skipping QC/QE plots (data missing or flag set)")

    if not args.skip_bsm and Path("data/BSM").exists():
        print("Rendering BSM plots…")
        bsm_data = read_bsm_data("data/BSM")
        bsm_outputs = plot_bsm_results_webpage(
            bsm_data,
            output_root=str(plots_root),
            file_format=args.format,
            dpi=args.dpi,
        )
        bsm_pair_bar_size = DEFAULT_BSM_CONFIG.get("typical_dataset_size", max(DEFAULT_BSM_CONFIG["train_sizes"]))
        bsm_loss_table = _loss_table(
            bsm_data[bsm_data["mass_a"] == "30"], model_order=DEFAULT_BSM_CONFIG["models"]
        )
        bsm_sic_bar_table = _sic_bar_table(
            bsm_data[bsm_data["mass_a"] == "30"],
            model_order=DEFAULT_BSM_CONFIG["models"],
            train_sizes=DEFAULT_BSM_CONFIG["train_sizes"],
            head_order=DEFAULT_BSM_CONFIG.get("heads"),
            target_train_size=max(DEFAULT_BSM_CONFIG["train_sizes"]),
        )
        bsm_sic_scatter_table = _sic_scatter_table(
            bsm_data[bsm_data["mass_a"] == "30"],
            model_order=DEFAULT_BSM_CONFIG["models"],
            train_sizes=DEFAULT_BSM_CONFIG["train_sizes"],
            head_order=DEFAULT_BSM_CONFIG.get("heads"),
        )
        bsm_pair_tables = [
            _scatter_table(
                bsm_data[bsm_data["mass_a"] == "30"],
                metric=DEFAULT_BSM_CONFIG["pair_scatter"]["metric"],
                train_sizes=DEFAULT_BSM_CONFIG["train_sizes"],
                model_order=DEFAULT_BSM_CONFIG["models"],
                head_order=DEFAULT_BSM_CONFIG.get("heads"),
            ),
            _bar_table(
                bsm_data[bsm_data["mass_a"] == "30"],
                metric=DEFAULT_BSM_CONFIG["pair_bar"]["metric"],
                model_order=DEFAULT_BSM_CONFIG["models"],
                head_order=DEFAULT_BSM_CONFIG.get("pair_heads") or [None],
                train_size_for_bar=bsm_pair_bar_size,
            ),
        ]
        bsm_sic_tables = [None, bsm_sic_bar_table, bsm_sic_scatter_table]
        bsm_plots = [
            {"src": str(Path(bsm_outputs["legend"]).relative_to(output_dir)), "caption": "BSM legend"},
            *[
                {
                    "src": str(Path(path).relative_to(output_dir)),
                    "caption": f"BSM: loss panel {i + 1}",
                    "table": bsm_loss_table,
                }
                for i, path in enumerate(bsm_outputs["loss"])
            ],
            *[
                {
                    "src": str(Path(path).relative_to(output_dir)),
                    "caption": f"BSM: pairing panel {i + 1}",
                    "table": bsm_pair_tables[i],
                }
                for i, path in enumerate(bsm_outputs["pair"])
            ],
            *[
                {
                    "src": str(Path(path).relative_to(output_dir)),
                    "caption": f"BSM: SIC panel {i + 1}",
                    "table": bsm_sic_tables[i],
                }
                for i, path in enumerate(bsm_outputs["sic"])
            ],
        ]
        sections.append(
            _section(
                "bsm",
                "Rare Higgs benchmark (H→2a→4b)",
                "Loss curves, pairing efficiency, and SIC summary for the BSM study.",
                bsm_plots,
            )
        )
    else:
        print("Skipping BSM plots (data missing or flag set)")

    if not args.skip_ad and Path("data/AD").exists():
        print("Rendering anomaly-detection plots…")
        ad_data = read_ad_data("data/AD")
        ad_outputs = plot_ad_results_webpage(
            ad_data,
            output_root=str(plots_root),
            file_format=args.format,
            dpi=args.dpi,
        )
        ad_sig_table = _ad_significance_table(ad_data["sig"])
        ad_gen_table = _ad_generation_table(ad_data["gen"], models_order=DEFAULT_AD_CONFIG["models"])
        ad_plots = [
            *[
                {
                    "src": str(Path(path).relative_to(output_dir)),
                    "caption": "AD: median significance by channel",
                    "table": ad_sig_table,
                }
                for path in ad_outputs["sig"]
            ],
            *[
                {
                    "src": str(Path(path).relative_to(output_dir)),
                    "caption": "AD: generative calibration and MMD",
                    "table": ad_gen_table,
                }
                for path in ad_outputs["gen"]
            ],
        ]
        sections.append(
            _section(
                "ad",
                "Dimuon anomaly-detection benchmark",
                "Significance summaries and calibration metrics across OS/SS channels.",
                ad_plots,
            )
        )
    else:
        print("Skipping anomaly-detection plots (data missing or flag set)")

    if not sections:
        raise SystemExit("No sections were rendered; check data availability or flags.")

    _render_index(output_dir, sections)


if __name__ == "__main__":
    main()
