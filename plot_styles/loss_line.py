"""
Backward-compatible entry point for loss plots.

This thin wrapper keeps the old import path alive while delegating to the
new modular implementation in :mod:`plot_styles.high_level.plot_loss`.
"""
from plot_styles.high_level.plot_loss import plot_loss  # re-export

__all__ = ["plot_loss"]
