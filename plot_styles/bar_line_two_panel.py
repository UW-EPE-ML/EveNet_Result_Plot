"""
Backward-compatible entry point for bar/line plots.

Delegates to :mod:`plot_styles.high_level.plot_bar_line`.
"""
from plot_styles.high_level.plot_bar_line import plot_bar_line  # re-export

__all__ = ["plot_bar_line"]
