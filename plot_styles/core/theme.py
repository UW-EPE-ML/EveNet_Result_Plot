from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Tuple
import matplotlib as mpl


@dataclass
class PlotStyle:
    """Lightweight style bundle for fonts and sizing.

    Attributes
    ----------
    base_font_size : float
        Overall font size baseline used for axis labels and text.
    axis_label_size : float | None
        Override for x/y label font size. When ``None`` the base is used.
    tick_label_size : float | None
        Override for tick label font size. Falls back to 90% of base.
    legend_size : float | None
        Override for legend text. Defaults to the tick label size.
    title_size : float | None
        Override for titles. Defaults to 105% of base.
    figure_scale : float
        Multiplier applied to figure dimensions via ``scaled_fig_size``.
    object_scale : float
        Multiplier for linewidths, markers, and tick lengths.
    """

    base_font_size: float = 13.0
    axis_label_size: float | None = None
    tick_label_size: float | None = None
    legend_size: float | None = None
    title_size: float | None = None
    figure_scale: float = 1.0
    object_scale: float = 1.0

    def _resolved_sizes(self) -> dict:
        tick_size = self.tick_label_size or self.base_font_size * 0.9
        legend_size = self.legend_size or tick_size
        return {
            "font.size": self.base_font_size,
            "axes.labelsize": self.axis_label_size or self.base_font_size,
            "axes.titlesize": self.title_size or self.base_font_size * 1.05,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "pdf.fonttype": 42,
        }

    @contextmanager
    def context(self):
        """Temporarily apply the style to matplotlib rcParams."""
        with mpl.rc_context(rc=self._resolved_sizes()):
            yield


def use_style(style: PlotStyle | None):
    """Return a context manager that applies ``style`` when provided."""
    return style.context() if style is not None else nullcontext()


def scaled_fig_size(fig_size: Tuple[float, float], *, scale: float = 1.0, aspect_ratio: float | None = None) -> Tuple[float, float]:
    """Scale a base figure size while optionally enforcing an aspect ratio.

    Parameters
    ----------
    fig_size : tuple
        Base ``(width, height)`` in inches.
    scale : float, optional
        Uniform multiplier applied to width and height.
    aspect_ratio : float, optional
        If provided, recompute height = width / aspect_ratio after scaling.
    """
    width, height = fig_size
    width *= scale
    height *= scale
    if aspect_ratio:
        height = width / aspect_ratio
    return width, height
