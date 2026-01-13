from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from plot_styles.core.legend import plot_legend
from plot_styles.core.style_axis import apply_nature_axis_style
from plot_styles.style import MODEL_COLORS

def boosted_alpha(mX, mY, mH=125.0, B0=1.5, B1=3.0):
    """
    Mass-only boostedness estimator for X -> Y H, Y -> bb.

    Parameters
    ----------
    mX : float or np.ndarray
        Mass of X (GeV)
    mY : float or np.ndarray
        Mass of Y (GeV)
    mH : float, optional
        Higgs mass (default = 125 GeV)
    B0 : float, optional
        Boost scale where alpha ~ 0.5 (default = 1.5)
    B1 : float, optional
        Saturation scale (default = 3.0)

    Returns
    -------
    alpha : float or np.ndarray
        Boostedness in [0, 1]
        0   -> fully resolved
        1   -> very boosted
    """

    mX = np.asarray(mX, dtype=float)
    mY = np.asarray(mY, dtype=float)

    # Kinematic threshold
    valid = mX > (mY + mH)

    # Two-body momentum of Y in X rest frame
    pY = np.zeros_like(mX)
    term1 = mX**2 - (mY + mH)**2
    term2 = mX**2 - (mY - mH)**2
    pY[valid] = np.sqrt(term1[valid] * term2[valid]) / (2.0 * mX[valid])

    # Boost proxy
    B = np.zeros_like(mX)
    B[valid] = pY[valid] / mY[valid]

    # Smooth mapping to [0,1]
    alpha = np.clip((B - B0) / (B1 - B0), 0.0, 1.0)

    return alpha

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


def draw_boosted_background(ax_list, ordered, cfg_boost):
    if not cfg_boost.get("enabled", False):
        return
    mH = cfg_boost.get("mH", 125.0)
    B0 = cfg_boost.get("B0", 1.5)
    B1 = cfg_boost.get("B1", 3.0)
    base_color = cfg_boost.get("base_color", "gray")
    alpha_min = cfg_boost.get("alpha_min", 0.0)
    alpha_max = cfg_boost.get("alpha_max", 0.6)
    alpha_scale = cfg_boost.get("alpha_scale", None)
    zorder = cfg_boost.get("zorder", 0.5)

    mx_vals = np.array([mx for mx, _ in ordered], dtype=float)
    my_vals = np.array([my for _, my in ordered], dtype=float)
    alpha_base = boosted_alpha(mx_vals, my_vals, mH=mH, B0=B0, B1=B1)
    if alpha_scale is not None:
        alpha_base = np.clip(alpha_base * alpha_scale, 0.0, 1.0)
    alphas = alpha_min + (alpha_max - alpha_min) * alpha_base

    for ax in ax_list:
        for idx, alpha in enumerate(alphas):
            if alpha <= 0.0:
                continue
            ax.axvspan(
                idx - 0.5,
                idx + 0.5,
                color=base_color,
                alpha=float(alpha),
                linewidth=0.0,
                zorder=zorder,
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
        cutflow_df=None,
        config=None,
        *,
        style=None,
):
    """Plot an unrolled grid with optional winner strip and ratio panels."""
    cfg = _merge_configs({}, config or {})
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
    cutflow_cfg = cfg.get("cutflow", {})
    cutflow_enabled = cutflow_cfg.get("enabled", False) and cutflow_df is not None

    nrows = 1 + (1 if winner_enabled else 0) + len(ratios_cfg)
    hr_main = cfg["height_ratios"]["main"]
    hr_winner = cfg["height_ratios"]["winner"]
    hr_ratio = cfg["height_ratios"]["ratio"]
    height_ratios = [hr_main]
    if winner_enabled:
        height_ratios.append(hr_winner)
    height_ratios += [hr_ratio] * len(ratios_cfg)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=cfg["figsize"],
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios, "hspace": cfg.get("hspace", 0.0)},
        constrained_layout=False
    )

    if nrows == 1:
        axes = [axes]

    n_points = len(x)
    for ax in axes:
        ax.set_xlim(-0.5, n_points - 0.5)

    ax_main = axes[0]
    ax_cutflow = ax_main.twinx() if cutflow_enabled else None
    ax_winner = axes[1] if winner_enabled else None
    ax_ratio_list = axes[(2 if winner_enabled else 1):]

    if cfg.get("y_main_log", False):
        ax_main.set_yscale("log")

    boost_axes = [ax_main]
    # if ax_winner is not None:
    #     boost_axes.append(ax_winner)
    boost_axes.extend(ax_ratio_list)
    draw_boosted_background(
        boost_axes,
        ordered,
        cfg.get("boost", {}),
    )

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

    if cutflow_enabled:
        round_digits = int(cutflow_cfg.get("round_digits", 6))
        passed_lookup = {
            (round(float(row["m_X"]), round_digits), round(float(row["m_Y"]), round_digits)): row["passed"]
            for _, row in cutflow_df.iterrows()
        }
        passed_vals = np.array(
            [
                passed_lookup.get(
                    (round(float(mx), round_digits), round(float(my), round_digits)),
                    np.nan,
                )
                for mx, my in ordered
            ],
            dtype=float,
        )
        ok = np.isfinite(passed_vals)
        ax_cutflow.patch.set_visible(False)
        ax_main.patch.set_alpha(0)
        ax_cutflow.set_zorder(ax_main.get_zorder() + 1)
        ax_cutflow.bar(
            x[ok],
            passed_vals[ok],
            width=cutflow_cfg.get("bar_width", 0.9),
            color=cutflow_cfg.get("color", "0.7"),
            edgecolor=cutflow_cfg.get("edgecolor", "0.2"),
            alpha=cutflow_cfg.get("alpha", 0.8),
            linewidth=cutflow_cfg.get("linewidth", 0.5),
            zorder=cutflow_cfg.get("zorder", 1),
        )
        ax_cutflow.set_ylabel(cutflow_cfg.get("ylabel", "Passed"), fontsize=label_fontsize_y)
        tick_fontsize_cutflow = cfg.get("tick_fontsize_cutflow", None)
        if tick_fontsize_cutflow is None and style is not None:
            tick_fontsize_cutflow = style.tick_label_size
        ax_cutflow.tick_params(axis="y", labelsize=tick_fontsize_cutflow)
        if cutflow_cfg.get("log", False):
            ax_cutflow.set_yscale("log")
            if np.any(ok):
                min_val = np.nanmin(passed_vals[ok])
                max_val = np.nanmax(passed_vals[ok])
                if min_val > 0:
                    ax_cutflow.set_ylim(min_val * 0.8, max_val * 1.2)
        elif np.any(ok):
            ax_cutflow.set_ylim(0.0, np.nanmax(passed_vals[ok]) * 1.1)

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
            axr.axhline(
                ref,
                color=cfg["block_separator"].get("color", "0.6"),
                linestyle=cfg["block_separator"].get("linestyle", "--"),
                linewidth=cfg["block_separator"].get("linewidth", 0.8),
            )

        ylabel = rcfg.get("ylabel", None)
        if ylabel is None:
            if mode == "ratio":
                ylabel = f"/ {base}"
            elif mode == "diff":
                ylabel = f"Δ vs {base}"
            else:
                ylabel = f"log / {base}"
        axr.set_ylabel(ylabel)
        if rcfg.get("y_log", False):
            axr.set_yscale("log")

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

    axis_list = [ax_main, *ax_ratio_list]
    _apply_axis_style(axis_list, cfg.get("apply_axis_style", False), style)

    plot_legend(
        fig,
        active_models=cfg['raw_model_series'],
        model_colors=MODEL_COLORS,
        legends=["models"],
        style=style,
        in_figure=True,
        y_start=1.01 if xb.get("show", True) else 1.01
    )


    fig.subplots_adjust(
        **cfg.get("subplot_adjust", {})
    )

    # plt.tight_layout(**cfg.get("tight_layout", {"pad": 0.2}))
    return fig, axes
