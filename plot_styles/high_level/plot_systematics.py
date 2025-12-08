import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from plot_styles.core.legend import add_ci_legend
from plot_styles.core.theme import PlotStyle, scaled_fig_size, use_style
from plot_styles.core.style_axis import apply_nature_axis_style
from plot_styles.style import MODEL_COLORS, MODEL_PRETTY
from mpl_toolkits.axes_grid1 import make_axes_locatable


def beeswarm_offsets(values: np.ndarray, max_width: float = 0.35, bins: int = 60) -> np.ndarray:
    """
    Deterministic KDE-based symmetric beeswarm jitter.
    Produces violin-shaped spread naturally, without random noise.

    ✓ Dense regions spread more
    ✓ Sparse regions stay narrow
    ✓ Symmetric jitter creates clean shape
    ✓ Fast for 1k+ points
    """
    x = np.asarray(values)

    # --- 1) KDE density estimate along x-axis ---
    kde = gaussian_kde(x)
    grid = np.linspace(x.min(), x.max(), bins)
    dens = kde(grid)
    dens = dens / dens.max()  # normalize to [0..1]

    # --- 2) Map each point to local density ---
    bin_idx = np.digitize(x, grid) - 1
    bin_idx = np.clip(bin_idx, 0, len(grid) - 1)
    local_width = dens[bin_idx] * max_width

    # --- 3) Symmetric jitter within local_width ---
    offsets = np.zeros_like(x, dtype=float)
    for b in np.unique(bin_idx):
        idx = np.where(bin_idx == b)[0]
        k = len(idx)
        if k > 1:
            # symmetric spiral: 0, +1, -1, +2, -2, ...
            order = np.argsort(x[idx])
            ranks = np.arange(k)
            jitter = ((ranks + 1) // 2) * ((-1) ** ranks)
            jitter = jitter / np.max(np.abs(jitter))  # normalize
            offsets[idx[order]] = jitter * local_width[idx]
        else:
            offsets[idx] = 0.0

    return offsets


def random_jitter(n, max_width=0.35, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-max_width, max_width, size=n)


def plot_systematic_scatter(
        data_df,
        *,
        model_order,
        metric_col: str,
        unc_col: str,
        color_col: str,
        fig_size=(8, 5.5),
        fig_scale: float = 1.0,
        fig_aspect: float | None = None,
        x_label: str = "",
        label: str | None = None,
        cmap: str = "coolwarm",
        colorbar_label: str = "",
        style: PlotStyle | None = None,
        with_legend: bool = True,
):
    """Plot a JES systematics beeswarm using a normalized metric."""

    with use_style(style):
        resolved_size = scaled_fig_size(fig_size, scale=fig_scale, aspect_ratio=fig_aspect)
        fig, ax = plt.subplots(figsize=resolved_size)

    active_models = []
    scale = style.object_scale if style else 1.0
    scatter = None

    for model in model_order:
        subset = data_df[data_df["model_pretty"] == model]
        if subset.empty:
            continue

        h_scale = 0.3
        position = len(active_models)
        active_models.append(model)

        x = subset[metric_col].to_numpy()
        scatter_y = random_jitter(len(x), max_width=0.3 * h_scale) + (position + 0.15)
        box_center = position - 0.2  # lower half band

        # Percentiles
        # box_color = MODEL_COLORS[model]
        # box_color = "#9F52C3"
        # box_color = "#AFA6C6"
        # ===== Compute statistics =====
        p50 = np.percentile(x, 50)  # median
        ci68_low, ci68_high = np.percentile(x, [16, 84])  # 68% central interval
        ci95_low, ci95_high = np.percentile(x, [2.5, 97.5])  # 95% central interval

        # ===== 95% band (lightest) =====
        col_outer = "#D9D9D9"
        ax.add_patch(plt.Rectangle(
            (ci95_low, box_center - h_scale * 0.50),  # x, y bottom
            ci95_high - ci95_low,  # width
            h_scale,  # height
            facecolor=col_outer,
            edgecolor=col_outer,
            alpha=0.65,
        ))

        # ===== 68% band (darker and inside the 95%) =====
        col_inner = "#A6A6A6"
        ax.add_patch(plt.Rectangle(
            (ci68_low, box_center - h_scale * 0.50),
            ci68_high - ci68_low,
            h_scale,
            facecolor=col_inner,
            edgecolor=col_inner,
            alpha=0.85,  # darker band
        ))

        # ===== Median line (strongest visual emphasis) =====
        ax.vlines(
            p50,
            box_center - h_scale * 0.50,
            box_center + h_scale * 0.50,
            color="black",
            linewidth=2.0 * scale,
            alpha=1.0
        )

        # Scatter cloud (upper layer)
        scatter = ax.scatter(
            x,
            scatter_y,
            c=subset[color_col],
            cmap=cmap,
            # s=10 * scale + subset[unc_col].to_numpy() * 60 * scale,
            s=25 * scale,
            alpha=0.95,
            linewidths=0,
            zorder=3,
        )

    ax.axvline(0, color="0.8", lw=1.5 * scale, ls="--", zorder=0)
    ax.set_yticks(range(len(active_models)))
    ax.set_yticklabels([MODEL_PRETTY[name] for name in active_models])
    ax.set_xlabel(x_label)

    xmin, xmax = ax.get_xlim()
    # ax.set_xlim(xmin - 1.0, xmax)

    apply_nature_axis_style(ax, style=style)
    if scatter is not None:
        # ---- Force global color range ----
        clim_min, clim_max = -2.0, 2.0
        scatter.set_clim(clim_min, clim_max)  # update the scatter normalization

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.1)  # size controls width
        cbar = plt.colorbar(scatter, cax=cax)

        # ---- Styling ----
        cbar.set_label(
            colorbar_label,
            labelpad=-0.5,
            fontsize=style.axis_label_size if style else None
        )
        cbar.ax.tick_params(labelsize=style.tick_label_size if style else None)

        # ---- Only two ticks (min & max) ----
        cbar.set_ticks([clim_min, clim_max])
        cbar.set_ticklabels([f"{clim_min:.0f}", f"{clim_max:.0f}"])
        cbar.ax.tick_params(size=0)  # hide tick marks
        cbar.ax.minorticks_off()

    if with_legend:
        add_ci_legend(
            fig,
            labels=["Median", "68% CI", "95% CI"],  # required 3 labels
            colors=["black", "#A6A6A6", "#D9D9D9"],  # median line, inner, outer
            style=style,  # << uses PlotStyle control
            horizontal=False,  # horizontal row legend
            frameon=False,
        )

    return fig, ax, active_models
