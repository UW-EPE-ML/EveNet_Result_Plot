import numpy as np
import matplotlib.pyplot as plt

from plot_styles.core.theme import PlotStyle, scaled_fig_size, use_style
from plot_styles.core.style_axis import apply_nature_axis_style


def _beeswarm_offsets(values: np.ndarray, max_width: float = 0.35, seed: int = 0) -> np.ndarray:
    """Return deterministic jitter offsets for a beeswarm effect."""

    if len(values) == 0:
        return np.array([])
    if len(values) == 1:
        return np.array([0.0])

    rng = np.random.default_rng(seed)
    try:
        density = np.asarray(values)
        density = density / (np.max(np.abs(density)) or 1.0)
    except Exception:
        density = np.ones_like(values, dtype=float)
    return rng.uniform(-1, 1, size=len(values)) * density * max_width


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

        position = len(active_models)
        active_models.append(model)
        x = subset[metric_col].to_numpy()
        y = _beeswarm_offsets(x, max_width=0.35 * scale) + position

        ci68 = np.percentile(x, [16, 84])
        ci95 = np.percentile(x, [2.5, 97.5])

        ax.fill_betweenx(
            [position - 0.25, position + 0.25],
            ci68[0],
            ci68[1],
            color="gray",
            alpha=0.06,
            linewidth=0,
            zorder=0,
        )

        ax.plot([ci95[0], ci95[1]], [position, position], color="0.75", lw=0.9 * scale, alpha=0.9, zorder=1)
        ax.scatter([ci95[0], ci95[1]], [position, position], color="0.55", s=10 * scale, alpha=0.9, zorder=2)

        scatter = ax.scatter(
            x,
            y,
            c=subset[color_col],
            cmap=cmap,
            s=18 * scale + subset[unc_col].to_numpy() * 60 * scale,
            alpha=0.9,
            linewidths=0,
            zorder=2,
        )

    ax.axvline(0, color="0.8", lw=0.8 * scale, ls="--", zorder=0)
    ax.set_yticks(range(len(active_models)))
    ax.set_yticklabels(active_models)
    ax.set_xlabel(x_label)

    apply_nature_axis_style(ax, style=style)
    if scatter is not None:
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label(colorbar_label, fontsize=style.axis_label_size if style else None)
        cbar.ax.tick_params(labelsize=style.tick_label_size if style else None)

    return fig, ax, active_models
