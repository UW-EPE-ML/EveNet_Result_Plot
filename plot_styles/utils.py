"""
Compatibility shims exposing the refreshed core utilities.

Existing modules import from ``plot_styles.utils``; this file now simply
re-exports the axis styling, legend, and save helpers from the new core
package to avoid duplication.
"""
from plot_styles.core.style_axis import apply_nature_axis_style, save_axis
from plot_styles.core.legend import plot_legend

__all__ = ["apply_nature_axis_style", "save_axis", "plot_legend"]
