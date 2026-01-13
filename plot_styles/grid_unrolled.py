from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from plot_styles.core.style_axis import apply_nature_axis_style

DEFAULT_UNROLLED_GRID_CONFIG = {
    "models": ["Pretrained", "Scratch", "XGBoost", "TabPFN"],
    # Layout
    "figsize": (15, 6),
    "hspace": 0.0,  # set to 0.0 to concatenate panels
    "height_ratios": {
        "main": 2.2,
        "winner": 0.35,
        "ratio": 1.0,
    },
    # Axes labels
    "x_top_label": r"$m_Y$ [GeV]",
    "x_bottom_label": r"$m_X$ [GeV]",
    "y_main_label": "SIC",
    # Ticks
    "tick_fontsize_top": 10,
    "tick_fontsize_bottom": 11,
    "tick_rotation_top": 90,
    "label_fontsize": None,
    "label_fontsize_top": None,
    "label_fontsize_bottom": None,
    "label_fontsize_y": None,
    "x_top": {
        "show": True,  # master switch for bottom axis handling
        "show_text": True,  # if False -> no text labels (best for many points)
        "show_ticks": True,  # if False -> remove ticks entirely
        "label_every": 2,  # show every N-th mY label if show_text=True
    },
    # Block separators (mX boundaries)
    "block_separator": {
        "enabled": True,
        "color": "0.6",
        "linestyle": "--",
        "linewidth": 0.8,
    },
    # Main lines
    "line": {
        "marker": "o",
        "markersize": 2.5,
        "linewidth": 1.2,
    },
    "legend_main": {
        "enabled": True,
        "ncols": 4,
        "fontsize": None,
        "loc": "upper left",
    },
    # Winner strip
    "winner": {
        "enabled": True,
        "ylabel": "Win",
        "ylabel_rotation": 90,
        "ylabel_pad": 38,
        "legend": {
            "enabled": True,
            "fontsize": 8,
            "ncols": 4,
            "bbox_to_anchor": (1.01, 0.5),
            "loc": "center left",
            "frameon": False,
        },
    },
    # Ratio panels (parameterized!)
    "ratios": [
        {
            "baseline": "XGBoost",
            "mode": "ratio",  # "ratio" | "diff" | "logratio"
            "ylabel": None,  # if None, auto-label based on mode & baseline
            "reference_line": True,
        },
        {
            "baseline": "TabPFN",
            "mode": "ratio",
            "ylabel": None,
            "reference_line": True,
        },
    ],
    "ratio_line": {
        "marker": "o",
        "markersize": 2.0,
        "linewidth": 1.0,
    },
    "unc": {
        "enabled": True,
        "models": ["Pretrained", "XGBoost"],  # empty list -> none shaded
        "alpha": 0.25,
        "zorder": 1,
    },
    "apply_axis_style": False,
    # Tight layout padding
    "tight_layout": {"pad": 0.2},
}


def _merge_configs(default: dict, override: dict | None) -> dict:
    if override is None:
        return default
    merged = {}
    for key, value in default.items():
        if key in override:
            if isinstance(value, dict) and isinstance(override[key], dict):
                merged[key] = _merge_configs(value, override[key])
            else:
                merged[key] = override[key]
        else:
            merged[key] = value
    for key, value in override.items():
        if key not in merged:
            merged[key] = value
    return merged


def build_unrolled_order(points):
    """Deterministic 1D order: sort by mX blocks, within each block by increasing mY."""
    mx_list = sorted({mx for mx, _ in points})
    ordered = []
    block_edges = [0]
    block_centers = []

    for mx in mx_list:
        start = len(ordered)
        mys = sorted(my for (mx2, my) in points if mx2 == mx)
        for my in mys:
            ordered.append((mx, my))
        end = len(ordered)
        block_edges.append(end)
        block_centers.append((start + end - 1) / 2.0)

    return ordered, block_edges, block_centers, mx_list


def series_from_dict(sic_by_model, models, ordered):
    """Map sic_by_model[model][(mx,my)] -> aligned arrays per model in 'ordered'."""
    y_vals = {}
    for model in models:
        model_map = sic_by_model[model]
        y_vals[model] = np.array([model_map[p] for p in ordered], dtype=float)
    return y_vals


def ratio_transform(y, y_base, mode="ratio", eps=1e-12):
    """Compute ratio-like transforms."""
    y = np.asarray(y, float)
    y_base = np.asarray(y_base, float)
    if mode == "diff":
        return y - y_base
    if mode == "logratio":
        return np.log((y + eps) / (y_base + eps))
    return (y + eps) / (y_base + eps)


def draw_block_separators(ax, block_edges, cfg_sep):
    if not cfg_sep.get("enabled", True):
        return
    for edge in block_edges:
        ax.axvline(
            edge - 0.5,
            color=cfg_sep.get("color", "0.6"),
            linestyle=cfg_sep.get("linestyle", "--"),
            linewidth=cfg_sep.get("linewidth", 0.8),
        )


def _apply_axis_style(ax_list, apply_style, style):
    if not apply_style:
        return
    for ax in ax_list:
        apply_nature_axis_style(ax, style=style)


def plot_unrolled_grid_with_winner_and_ratios(
    points,
    sic_by_model,
    unc_by_model=None,
    config=None,
    *,
    style=None,
):
    """Plot an unrolled grid with optional winner strip and ratio panels."""
    cfg = _merge_configs(DEFAULT_UNROLLED_GRID_CONFIG, config or {})
    style = cfg.get("style", style)
    label_fontsize = cfg.get("label_fontsize", None)
    label_fontsize_y = cfg.get("label_fontsize_y", label_fontsize)
    label_fontsize_top = cfg.get("label_fontsize_top", label_fontsize)
    label_fontsize_bottom = cfg.get("label_fontsize_bottom", label_fontsize)
    if style is not None:
        if label_fontsize is None:
            label_fontsize = style.axis_label_size
        if label_fontsize_y is None:
            label_fontsize_y = style.axis_label_size
        if label_fontsize_top is None:
            label_fontsize_top = style.axis_label_size
        if label_fontsize_bottom is None:
            label_fontsize_bottom = style.axis_label_size

    models = cfg["models"]
    ordered, block_edges, block_centers, mx_list = build_unrolled_order(points)
    x = np.arange(len(ordered))

    y_vals = series_from_dict(sic_by_model, models, ordered)

    ratios_cfg = cfg.get("ratios", [])
    winner_enabled = cfg.get("winner", {}).get("enabled", True)

    nrows = 1 + (1 if winner_enabled else 0) + len(ratios_cfg)
    hr_main = cfg["height_ratios"]["main"]
    hr_winner = cfg["height_ratios"]["winner"]
    hr_ratio = cfg["height_ratios"]["ratio"]
    height_ratios = [hr_main] + ([hr_winner] if winner_enabled else []) + [hr_ratio] * len(ratios_cfg)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=cfg["figsize"],
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios, "hspace": cfg.get("hspace", 0.0)},
    )

    if nrows == 1:
        axes = [axes]

    n_points = len(x)
    for ax in axes:
        ax.set_xlim(-0.5, n_points - 0.5)

    ax_main = axes[0]
    ax_winner = axes[1] if winner_enabled else None
    ax_ratio_list = axes[(2 if winner_enabled else 1):]

    line_cfg = cfg["line"]
    color_map = cfg.get("color_map") or line_cfg.get("colors") or {}
    line_handles = []
    model_colors = {}

    for model in models:
        color = color_map.get(model)
        (line_handle,) = ax_main.plot(
            x,
            y_vals[model],
            marker=line_cfg.get("marker", "o"),
            markersize=line_cfg.get("markersize", 2.5),
            linewidth=line_cfg.get("linewidth", 1.2),
            label=model,
            color=color,
        )
        model_colors[model] = line_handle.get_color()

        unc_cfg = cfg.get("unc", {})
        if (
            unc_cfg.get("enabled", True)
            and unc_by_model is not None
            and model in set(unc_cfg.get("models", []))
            and model in unc_by_model
        ):
            sigma = np.array([unc_by_model[model].get(p, np.nan) for p in ordered], dtype=float)
            ok = np.isfinite(sigma)
            if np.any(ok):
                ax_main.fill_between(
                    x[ok],
                    (y_vals[model][ok] - sigma[ok]),
                    (y_vals[model][ok] + sigma[ok]),
                    color=model_colors[model],
                    alpha=unc_cfg.get("alpha", 0.25),
                    linewidth=0.0,
                    zorder=unc_cfg.get("zorder", 1),
                )

        line_handles.append(line_handle)

    ax_main.set_ylabel(cfg.get("y_main_label", "SIC"), fontsize=label_fontsize_y)

    if cfg.get("legend_main", {}).get("enabled", True):
        leg_cfg = cfg["legend_main"]
        handles = [Patch(facecolor=model_colors[m], edgecolor="none", label=m) for m in models]
        legend_fontsize = leg_cfg.get("fontsize")
        if legend_fontsize is None and style is not None:
            legend_fontsize = style.legend_size
        ax_main.legend(
            handles=handles,
            ncols=leg_cfg.get("ncols", 4),
            fontsize=legend_fontsize,
            loc=leg_cfg.get("loc", "upper left"),
            frameon=leg_cfg.get("frameon", False),
        )

    draw_block_separators(ax_main, block_edges, cfg["block_separator"])

    ax_top = ax_main.twiny()
    ax_top.set_xlim(ax_main.get_xlim())
    ax_top.set_xticks(x)

    xb = cfg.get("x_top", {})
    if xb.get("show", True):
        ax_top.set_xlabel(
            cfg.get("x_top_label", r"$m_Y$ [GeV] (per point)"),
            fontsize=label_fontsize_top,
        )

        if xb.get("show_ticks", True):
            if xb.get("show_text", True):
                label_every = int(xb.get("label_every", 1))
                full_labels = [f"{int(my)}" for (_, my) in ordered]
                labels = [lab if (i % label_every == 0) else "" for i, lab in enumerate(full_labels)]
                tick_fontsize_top = cfg.get("tick_fontsize_top", 11)
                if tick_fontsize_top is None and style is not None:
                    tick_fontsize_top = style.tick_label_size
                ax_top.set_xticklabels(
                    labels,
                    fontsize=tick_fontsize_top,
                    rotation=cfg.get("tick_rotation_top", 90),
                )
            else:
                ax_top.set_xticklabels([])
                ax_top.tick_params(axis="x", which="both", length=3)
        else:
            ax_top.set_xticks([])
            ax_top.set_xticklabels([])
    else:
        ax_top.set_xlabel("")
        ax_top.set_xticks([])
        ax_top.set_xticklabels([])

    if winner_enabled:
        stack = np.vstack([np.nan_to_num(y_vals[m], nan=-np.inf) for m in models])
        winner_idx = np.argmax(stack, axis=0)
        cmap = ListedColormap([model_colors[m] for m in models])

        ax_winner.imshow(
            winner_idx[np.newaxis, :],
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=-0.5,
            vmax=len(models) - 0.5,
        )
        ax_winner.set_yticks([])
        win_cfg = cfg["winner"]
        ax_winner.set_ylabel(
            win_cfg.get("ylabel", "win"),
            rotation=win_cfg.get("ylabel_rotation", 0),
            labelpad=win_cfg.get("ylabel_pad", 15),
            va="center",
        )

        draw_block_separators(ax_winner, block_edges, cfg["block_separator"])

    rline_cfg = cfg["ratio_line"]
    for axr, rcfg in zip(ax_ratio_list, ratios_cfg):
        base = rcfg["baseline"]
        mode = rcfg.get("mode", "ratio")
        y_base = y_vals[base]

        for model in models:
            if model == base:
                continue
            axr.plot(
                x,
                ratio_transform(y_vals[model], y_base, mode=mode),
                marker=rline_cfg.get("marker", "o"),
                markersize=rline_cfg.get("markersize", 2.0),
                linewidth=rline_cfg.get("linewidth", 1.0),
                color=model_colors[model],
            )

        if rcfg.get("reference_line", True):
            ref = 1.0 if mode == "ratio" else 0.0
            axr.axhline(ref, linewidth=0.8)

        ylabel = rcfg.get("ylabel", None)
        if ylabel is None:
            if mode == "ratio":
                ylabel = f"/ {base}"
            elif mode == "diff":
                ylabel = f"Δ vs {base}"
            else:
                ylabel = f"log / {base}"
        axr.set_ylabel(ylabel)

        draw_block_separators(axr, block_edges, cfg["block_separator"])

    for a in axes[:-1]:
        a.tick_params(labelbottom=False)

    axes[-1].set_xticks(block_centers)
    tick_fontsize_bottom = cfg.get("tick_fontsize_bottom", 10)
    if tick_fontsize_bottom is None and style is not None:
        tick_fontsize_bottom = style.tick_label_size
    axes[-1].set_xticklabels(
        [f"{int(mx)}" for mx in mx_list],
        fontsize=tick_fontsize_bottom,
    )
    axes[-1].set_xlabel(
        cfg.get("x_bottom_label", r"$m_X$ [GeV]"),
        fontsize=label_fontsize_bottom,
    )

    _apply_axis_style([ax_main, *ax_ratio_list], cfg.get("apply_axis_style", False), style)

    plt.tight_layout(**cfg.get("tight_layout", {"pad": 0.2}))
    return fig, axes
