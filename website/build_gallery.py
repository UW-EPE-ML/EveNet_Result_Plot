"""Build a static plot gallery for GitHub Pages.

This script regenerates PNG renderings of the paper plots from the
repository data and assembles a simple static HTML gallery. It is used by
CI to publish the figures to GitHub Pages, but can also be run locally
for spot-checking:

    python website/build_gallery.py --output-dir site --format png --dpi 200
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

# Use a headless backend for CI and local builds without a display
matplotlib.use("Agg")

from paper_plot import (  # noqa: E402  # pylint: disable=wrong-import-position
    read_ad_data,
    read_bsm_data,
    read_qe_data,
    plot_ad_results,
    plot_bsm_results,
    plot_qe_results,
)


def _render_index(output_dir: Path, sections: List[Dict[str, str]]) -> None:
    """Write a minimal HTML page to preview all generated plots."""

    cards = []
    for section in sections:
        card_items = "\n".join(
            [
                f"<figure><img src=\"{plot['src']}\" alt=\"{plot['caption']}\" "
                f"loading=\"lazy\"><figcaption>{plot['caption']}</figcaption></figure>"
                for plot in section["plots"]
            ]
        )
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
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 1rem;
      align-items: start;
    }}
    figure {{
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 8px;
      padding: 0.5rem;
      margin: 0;
      box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
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
    plots: List[Dict[str, str]],
) -> Dict[str, str]:
    return {
        "id": section_id,
        "title": title,
        "blurb": blurb,
        "plots": plots,
    }


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

    sections: List[Dict[str, str]] = []

    if not args.skip_qe and Path("data/QE_results_table.csv").is_file():
        print("Rendering QC/QE plots…")
        qe_data = read_qe_data("data/QE_results_table.csv")
        plot_qe_results(
            qe_data,
            output_root=str(plots_root),
            file_format=args.format,
            dpi=args.dpi,
            save_individual_axes=False,
        )
        sections.append(
            _section(
                "qe",
                "Quantum-correlation benchmark (QC)",
                "Validation loss, pairing efficiency, and DeltaD precision across training sizes.",
                [
                    {"src": f"plots/QE/loss.{args.format}", "caption": "QC: validation loss vs. effective steps"},
                    {"src": f"plots/QE/pair.{args.format}", "caption": "QC: pairing efficiency across datasets"},
                    {"src": f"plots/QE/deltaD.{args.format}", "caption": "QC: DeltaD precision vs. train size"},
                ],
            )
        )
    else:
        print("Skipping QC/QE plots (data missing or flag set)")

    if not args.skip_bsm and Path("data/BSM").exists():
        print("Rendering BSM plots…")
        bsm_data = read_bsm_data("data/BSM")
        plot_bsm_results(
            bsm_data,
            output_root=str(plots_root),
            file_format=args.format,
            dpi=args.dpi,
            save_individual_axes=False,
        )
        sections.append(
            _section(
                "bsm",
                "Rare Higgs benchmark (H→2a→4b)",
                "Loss curves, pairing efficiency, and SIC summary for the BSM study.",
                [
                    {"src": f"plots/BSM/loss.{args.format}", "caption": "BSM: validation loss per model/head"},
                    {"src": f"plots/BSM/pair.{args.format}", "caption": "BSM: pairing efficiency by dataset size"},
                    {"src": f"plots/BSM/sic.{args.format}", "caption": "BSM: signal significance (SIC) summary"},
                ],
            )
        )
    else:
        print("Skipping BSM plots (data missing or flag set)")

    if not args.skip_ad and Path("data/AD").exists():
        print("Rendering anomaly-detection plots…")
        ad_data = read_ad_data("data/AD")
        plot_ad_results(
            ad_data,
            output_root=str(plots_root),
            file_format=args.format,
            dpi=args.dpi,
            save_individual_axes=False,
        )
        sections.append(
            _section(
                "ad",
                "Dimuon anomaly-detection benchmark",
                "Significance summaries and calibration metrics across OS/SS channels.",
                [
                    {"src": f"plots/AD/ad_significance.{args.format}", "caption": "AD: median significance by channel"},
                    {"src": f"plots/AD/ad_generation.{args.format}", "caption": "AD: generative calibration and MMD"},
                ],
            )
        )
    else:
        print("Skipping anomaly-detection plots (data missing or flag set)")

    if not sections:
        raise SystemExit("No sections were rendered; check data availability or flags.")

    _render_index(output_dir, sections)


if __name__ == "__main__":
    main()
