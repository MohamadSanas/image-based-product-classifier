# ============================================================
# Package : src/reporting
# Owner   : AHAMED A.A.
# Course  : EC9570 – Digital Image Processing
# Desc    : Visualization and statistical analysis module.
#           Generates bar charts, pie charts, and structured
#           statistical reports from ClassificationResult data.
# ============================================================

from .visualizer import ProductVisualizer
from .stats import compute_stats, format_stats_report

__all__ = [
    "ProductVisualizer",
    "compute_stats",
    "format_stats_report",
]
