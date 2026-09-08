# ============================================================
# src/detection/__init__.py
# Exposes the public API of the detection module.
# ============================================================

from .detector import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    Detection,
    ProductDetector,
)

__all__ = [
    "ProductDetector",
    "Detection",
    "DEFAULT_CONF_THRESHOLD",
    "DEFAULT_IOU_THRESHOLD",
]
