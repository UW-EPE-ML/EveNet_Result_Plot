from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

from plot_styles.core.style_axis import apply_nature_axis_style
from plot_styles.core.theme import PlotStyle, scaled_fig_size, use_style
from plot_styles.style import AD_MASS_FPR_COLORS, AD_MASS_FPR_ORDER

BIN_PERCENT = {8: 2.3, 12: 1.5, 16: 1.1}
FIT_TYPE = {3: "Cubic", 5: "Quintic", 7: "Septic"}

REQUIRED_KEYS = {
    "fpr_thresholds",
    "fit_degree",
    "num_bins_SR",
    "SB_left",
    "SR_left",
    "SR_right",
    "SB_right",
    "popts",
    "pcovs",
    "significances",
    "filtered_masses",
    "y_vals",
    "plot_bins_all",
    "plot_centers_all",
    "plot_centers_SB",
    "channel",
}


def _is_save_data(obj: Any) -> bool:
    return isinstance(obj, dict) and all(k in obj for k in REQUIRED_KEYS)


def _maybe_item(arr: Any) -> Any:
    if isinstance(arr, np.ndarray) and arr.shape == ():
        try:
            return arr.item()
        except Exception:
            return arr
    return arr


def _coerce_np(arr: Any, dtype=float) -> np.ndarray:
    return np.asarray(arr, dtype=dtype)


def _find_save_data(obj: Any) -> dict[str, Any] | None:
    if isinstance(obj, dict):
        if _is_save_data(obj):
            return obj
        if "save_data" in obj and isinstance(obj["save_data"], dict):
            found = _find_save_data(obj["save_data"])
            if found is not None:
                return found
        for value in obj.values():
            found = _find_save_data(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_save_data(value)
            if found is not None:
                return found
    return None


def _find_bonus_significance(obj: Any) -> float | None:
    if isinstance(obj, dict):
        for key in ("bonus_significance", "q"):
            if key in obj:
                try:
                    return float(obj[key])
                except Exception:
                    pass
        for value in obj.values():
            found = _find_bonus_significance(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_bonus_significance(value)
            if found is not None:
                return found
    return None


def _load_npz_payload(npz_path: str | Path) -> dict[str, Any]:
    loaded = np.load(npz_path, allow_pickle=True)

    if "payload" in loaded.files:
        payload = _maybe_item(loaded["payload"])
        if not isinstance(payload, dict):
            raise RuntimeError("npz['payload'] is present but is not a dict.")
        return payload

    if "save_data" in loaded.files:
        payload: dict[str, Any] = {
            "save_data": _maybe_item(loaded["save_data"]),
        }
        if "bonus_significance" in loaded.files:
            payload["bonus_significance"] = float(_maybe_item(loaded["bonus_significance"]))
        elif "q" in loaded.files:
            payload["q"] = float(_maybe_item(loaded["q"]))
        return payload

    return {k: _maybe_item(loaded[k]) for k in loaded.files}


def _poly_fit(x: np.ndarray, *coeffs: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    for degree, coeff in enumerate(coeffs):
        y += float(coeff) * (x ** degree)
    return y


def _resolve_threshold_indices(available: Sequence[float], requested: Sequence[float] | None) -> list[int]:
    if requested is None:
        return list(range(len(available)))

    indices: list[int] = []
    for threshold in requested:
        match = next(
            (i for i, candidate in enumerate(available) if
             np.isclose(candidate, float(threshold), rtol=1e-9, atol=1e-12)),
            None,
        )
        if match is not None and match not in indices:
            indices.append(match)

    return indices if indices else list(range(len(available)))


def _format_threshold_label(threshold: float) -> str:
    percentage = 100.0 * threshold
    percentage_text = f"{percentage:.3g}"
    return f"FPR ≤ {percentage_text}%"


def _resolve_threshold_color(threshold: float, fallback_idx: int) -> str:
    for candidate in AD_MASS_FPR_ORDER:
        if np.isclose(float(candidate), float(threshold), rtol=1e-9, atol=1e-12):
            return AD_MASS_FPR_COLORS.get(candidate, "#000000")
    fallback = [AD_MASS_FPR_COLORS[x] for x in AD_MASS_FPR_ORDER if x in AD_MASS_FPR_COLORS]
    if fallback:
        return fallback[fallback_idx % len(fallback)]
    return "#000000"


def _plot_upsilon_lines(
        ax,
        *,
        y_fraction: float = 0.96,
        with_labels: bool = True,
        line_color: str = "0.35",
        line_alpha: float = 0.32,
        line_lw: float = 1.4,
        line_ls: str = "--",
):
    resonances = [
        (9.460, r"$\Upsilon(1S)$"),
        (10.023, r"$\Upsilon(2S)$"),
        (10.355, r"$\Upsilon(3S)$"),
    ]
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for mass, label in resonances:
        ax.axvline(mass, color=line_color, linestyle=line_ls, alpha=line_alpha, lw=line_lw, zorder=2)
        if with_labels:
            ax.text(
                mass * 0.995,
                y_fraction,
                label,
                rotation=90,
                verticalalignment="center",
                horizontalalignment="right",
                transform=trans,
            )


def load_ad_mass_payload(npz_path: str | Path) -> dict[str, Any]:
    npz_path = Path(npz_path)
    payload = _load_npz_payload(npz_path)
    save_data = _find_save_data(payload)
    if save_data is None:
        raise RuntimeError(f"Could not find AD mass-distribution save_data in {npz_path}.")

    return {
        "npz_path": str(npz_path),
        "payload": payload,
        "save_data": save_data,
        "bonus_significance": _find_bonus_significance(payload),
    }


def plot_ad_mass_distribution(
        mass_payload: Mapping[str, Any],
        *,
        fpr_thresholds: Sequence[float] | None = None,
        fig_size: tuple[float, float] = (12.0, 9.0),
        fig_scale: float = 1.0,
        fig_aspect: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        x_min: float | None = None,
        x_max: float | None = None,
        cms_label: str | None = "2016 CMS Open Data DoubleMuon",
        legend_loc: str | tuple[float, float] = (0.62, 0.68),
        legend_font_size: float | None = None,
        legend_ncols: int = 1,
        legend_bbox_to_anchor: tuple[float, float] | None = None,
        legend_marker_size: float = 10.5,
        legend_handlelength: float = 1.8,
        legend_labelspacing: float = 0.35,
        legend_borderaxespad: float = 0.25,
        legend_columnspacing: float = 0.9,
        cms_label_size: float | None = None,
        cms_label_weight: str = "bold",
        cms_label_x: float = 0.01,
        cms_label_y: float = 0.99,
        cme_lumi_label: str | None = r"$8.7\ \mathrm{fb}^{-1},\ \sqrt{s}=13\ \mathrm{TeV}$",
        cme_lumi_size: float | None = None,
        cme_lumi_x: float = 0.99,
        cme_lumi_y: float = 0.99,
        cme_lumi_style: str = "italic",
        cme_lumi_weight: str = "normal",
        text_block_alpha: float = 0.85,
        text_font_size: float | None = None,
        text_line_spacing: float = 0.043,
        text_x: float = 0.05,
        text_y: float = 0.95,
        lower_panel: bool = False,
        lower_panel_fpr_thresholds: Sequence[float] | None = None,
        lower_panel_height_ratio: tuple[float, float] = (3.4, 1.25),
        lower_panel_label: str = "Data - Predict",
        lower_panel_y_min: float | None = None,
        lower_panel_y_max: float | None = None,
        lower_panel_nbins: int = 5,
        lower_panel_show_markers: bool = False,
        show_upsilon: bool = True,
        show_fit_band: bool = True,
        fit_band_samples: int = 200,
        f_name: str | None = None,
        plot_dir: str = ".",
        file_format: str | None = None,
        dpi: int | None = None,
        style: PlotStyle | None = None,
) -> dict[str, Any]:
    save_data_obj = mass_payload.get("save_data", mass_payload)
    if not isinstance(save_data_obj, dict):
        raise RuntimeError("Mass payload must include a dict-like save_data.")

    save_data = save_data_obj
    npz_path = mass_payload.get("npz_path")
    bonus_significance = mass_payload.get("bonus_significance", None)

    thresholds = [float(x) for x in save_data["fpr_thresholds"]]
    selected_indices = _resolve_threshold_indices(thresholds, fpr_thresholds)

    fit_degree = int(save_data["fit_degree"])
    num_bins_sr = int(save_data["num_bins_SR"])
    sb_left = float(save_data["SB_left"])
    sr_left = float(save_data["SR_left"])
    sr_right = float(save_data["SR_right"])
    sb_right = float(save_data["SB_right"])
    channel = str(save_data["channel"])

    plot_bins_all = _coerce_np(save_data["plot_bins_all"], dtype=float)
    plot_centers_all = _coerce_np(save_data["plot_centers_all"], dtype=float)
    plot_centers_sb = _coerce_np(save_data["plot_centers_SB"], dtype=float)

    popts = save_data["popts"]
    pcovs = save_data["pcovs"]
    significances = [float(v) for v in save_data["significances"]]
    filtered_masses = save_data["filtered_masses"]
    y_vals_list = save_data["y_vals"]

    resolved_y_min = float(y_min if y_min is not None else save_data.get("ymin", 1e-2))
    resolved_y_max = float(y_max if y_max is not None else save_data.get("ymax", 1e5))
    resolved_x_min = float(x_min if x_min is not None else sb_left)
    resolved_x_max = float(x_max if x_max is not None else sb_right)

    lower_panel_indices: list[int] = []
    if lower_panel:
        if lower_panel_fpr_thresholds is None:
            lower_panel_indices = list(selected_indices)
        else:
            requested_indices = _resolve_threshold_indices(thresholds, lower_panel_fpr_thresholds)
            requested_set = set(requested_indices)
            lower_panel_indices = [idx for idx in selected_indices if idx in requested_set]
            if not lower_panel_indices:
                lower_panel_indices = list(selected_indices)
    lower_panel_index_set = set(lower_panel_indices)

    with use_style(style):
        figure_size = scaled_fig_size(fig_size, scale=fig_scale, aspect_ratio=fig_aspect)
        if lower_panel:
            fig, (ax_main, ax_lower) = plt.subplots(
                2,
                1,
                figsize=figure_size,
                sharex=True,
                gridspec_kw={
                    "height_ratios": [float(lower_panel_height_ratio[0]), float(lower_panel_height_ratio[1])],
                    "hspace": 0.0,
                },
            )
        else:
            fig, ax_main = plt.subplots(figsize=figure_size)
            ax_lower = None

    resolved_text_size = text_font_size
    if resolved_text_size is None and style is not None and style.tick_label_size is not None:
        resolved_text_size = max(10.0, float(style.tick_label_size) - 1.5)

    threshold_rows = []
    legend_entries = []

    for color_idx, idx in enumerate(selected_indices):
        threshold = thresholds[idx]
        popt = _coerce_np(popts[idx], dtype=float)
        masses = _coerce_np(filtered_masses[idx], dtype=float)
        y_vals = _coerce_np(y_vals_list[idx], dtype=float)
        significance = significances[idx]
        color = _resolve_threshold_color(threshold, color_idx)

        label = _format_threshold_label(threshold)

        prediction_all = _poly_fit(plot_centers_all, *popt)
        ax_main.plot(plot_centers_all, prediction_all, lw=2.2, linestyle="--", color=color)

        if show_fit_band:
            try:
                pcov = _coerce_np(pcovs[idx], dtype=float)
                if pcov.ndim == 2 and np.all(np.isfinite(pcov)):
                    rng = np.random.default_rng(42 + idx)
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=r".*encountered in matmul.*",
                            category=RuntimeWarning,
                        )
                        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                            sampled = rng.multivariate_normal(
                                popt,
                                pcov,
                                size=max(50, int(fit_band_samples)),
                                check_valid="ignore",
                            )
                            y_band = np.asarray(
                                [_poly_fit(plot_centers_all, *params) for params in sampled],
                                dtype=float,
                            )
                    y_low, y_high = np.nanpercentile(y_band, [16, 84], axis=0)
                    ax_main.fill_between(plot_centers_all, y_low, y_high, color=color, alpha=0.16)
            except Exception:
                pass

        ax_main.hist(
            masses,
            bins=plot_bins_all,
            histtype="step",
            linewidth=2.4,
            color=color,
            label=label,
            alpha=0.9,
        )

        if ax_lower is not None and idx in lower_panel_index_set:
            data_counts, _ = np.histogram(masses, bins=plot_bins_all)
            residual = data_counts.astype(float) - prediction_all
            residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
            ax_lower.stairs(
                residual,
                plot_bins_all,
                color=color,
                linewidth=2.0,
                baseline=0.0,
                alpha=0.95,
            )

        threshold_rows.append(
            {
                "fpr_threshold": threshold,
                "significance": significance,
                "selected_events": int(masses.shape[0]),
            }
        )
        legend_entries.append(
            {
                "label": label,
                "color": color,
                "linestyle": "-",
            }
        )

    channel_label = "Opposite Sign" if channel.upper() == "OS" else "Same Sign" if channel.upper() == "SS" else channel
    text_lines: list[str] = []
    if text_lines:
        x0 = text_x
        y0 = text_y
        dy = text_line_spacing
        for idx, line in enumerate(text_lines):
            ax_main.text(
                x0,
                y0 - idx * dy,
                line,
                transform=ax_main.transAxes,
                va="top",
                alpha=text_block_alpha,
                fontsize=resolved_text_size,
                linespacing=0.95,
            )

    for axis in [ax_main, ax_lower] if ax_lower is not None else [ax_main]:
        axis.axvline(sr_left, color="0.35", lw=1.4, linestyle="--", alpha=0.65, zorder=7)
        axis.axvline(sr_right, color="0.35", lw=1.4, linestyle="--", alpha=0.65, zorder=7)

    if ax_lower is not None:
        ax_main.set_xlabel("")
        ax_main.tick_params(labelbottom=False)
        ax_lower.set_xlabel(r"$M_{\mu\mu}$ [GeV]")
    else:
        ax_main.set_xlabel(r"$M_{\mu\mu}$ [GeV]")

    ax_main.set_ylabel("Events")
    ax_main.set_yscale("log")
    ax_main.set_ylim(resolved_y_min, resolved_y_max)
    ax_main.set_xlim(resolved_x_min, resolved_x_max)

    apply_nature_axis_style(ax_main, style=style)
    ax_main.minorticks_on()
    ax_main.tick_params(axis="x", which="minor", bottom=True)
    ax_main.tick_params(axis="y", which="minor", left=True)
    if ax_lower is not None:
        ax_main.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    if ax_lower is not None:
        ax_lower.set_ylabel(lower_panel_label)
        apply_nature_axis_style(ax_lower, style=style)
        ax_lower.axhline(0.0, color="0.35", linewidth=1.2, linestyle="--", alpha=0.55, zorder=1)
        if lower_panel_y_min is not None:
            ax_lower.set_ylim(bottom=lower_panel_y_min)
        if lower_panel_y_max is not None:
            ax_lower.set_ylim(top=lower_panel_y_max)
        ax_lower.yaxis.set_major_locator(MaxNLocator(nbins=max(1, int(lower_panel_nbins))))
        ax_lower.minorticks_on()
        ax_lower.tick_params(axis="x", which="minor", bottom=True)
        ax_lower.tick_params(axis="y", which="minor", left=True)

    if show_upsilon:
        _plot_upsilon_lines(ax_main, y_fraction=0.92, with_labels=True)
        if ax_lower is not None:
            _plot_upsilon_lines(ax_lower, with_labels=False)

    if legend_entries:
        resolved_legend_size = (
            legend_font_size
            if legend_font_size is not None
            else (
                style.legend_size
                if style is not None and style.legend_size is not None
                else None
            )
        )
        if resolved_legend_size is None:
            resolved_legend_size = 13.0

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                markersize=legend_marker_size,
                linestyle="",
                color=entry["color"],
                markeredgecolor="black",
                label=entry["label"],
            )
            for entry in legend_entries
        ]
        legend_handles.extend([
            Line2D([0], [0], color="black", linewidth=2.0, linestyle="-", label="Data"),
            Line2D([0], [0], color="black", linewidth=2.0, linestyle="--", label="Predict (Bkg fit)"),
        ])
        ax_main.legend(
            handles=legend_handles,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            frameon=False,
            fontsize=resolved_legend_size,
            ncol=max(1, int(legend_ncols)),
            handlelength=legend_handlelength,
            handletextpad=0.4,
            labelspacing=legend_labelspacing,
            borderaxespad=legend_borderaxespad,
            columnspacing=legend_columnspacing,
        )

    if cms_label:
        resolved_cms_size = (
            cms_label_size
            if cms_label_size is not None
            else (
                style.cms_label_fontsize
                if style is not None and style.cms_label_fontsize is not None
                else max((legend_font_size if legend_font_size is not None else 13.0) + 2.0, 14.0)
            )
        )
        ax_main.text(
            cms_label_x,
            cms_label_y,
            cms_label,
            transform=ax_main.transAxes,
            ha="left",
            va="bottom",
            fontsize=resolved_cms_size,
            fontweight=cms_label_weight,
        )
    if cme_lumi_label:
        resolved_cme_lumi_size = (
            cme_lumi_size
            if cme_lumi_size is not None
            else (
                (legend_font_size - 1.0)
                if legend_font_size is not None
                else (
                    (style.legend_size - 1.0)
                    if style is not None and style.legend_size is not None
                    else 12.0
                )
            )
        )
        ax_main.text(
            cme_lumi_x,
            cme_lumi_y,
            cme_lumi_label,
            transform=ax_main.transAxes,
            ha="right",
            va="bottom",
            fontsize=resolved_cme_lumi_size,
            fontstyle=cme_lumi_style,
            fontweight=cme_lumi_weight,
        )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    def _with_ext(name: str) -> str:
        root, ext = os.path.splitext(name)
        if ext:
            return name
        if file_format:
            return f"{root}.{file_format}"
        return f"{root}.pdf"

    output_path = None
    if f_name is not None:
        os.makedirs(plot_dir, exist_ok=True)
        output_path = os.path.join(plot_dir, _with_ext(f_name))
        fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
        print(f"Saved figure → {output_path}")

    return {
        "npz_path": npz_path,
        "output_path": output_path,
        "channel": channel,
        "channel_label": channel_label,
        "fit_degree": fit_degree,
        "fit_name": FIT_TYPE.get(fit_degree, str(fit_degree)),
        "num_bins_sr": num_bins_sr,
        "sb_left": sb_left,
        "sr_left": sr_left,
        "sr_right": sr_right,
        "sb_right": sb_right,
        "bonus_significance": None if bonus_significance is None else float(bonus_significance),
        "threshold_rows": threshold_rows,
        "lower_panel_fpr_thresholds": [thresholds[idx] for idx in lower_panel_indices],
    }
