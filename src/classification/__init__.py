# ============================================================
# src/classification/__init__.py
# Exposes the public API of the classification module.
# ============================================================

from .class_map import (
    RPC_CATEGORIES,
    RPC_CATEGORY_MAP,
    RPC_CLASS_NAMES,
    get_category,
)
from .classifier import (
    DEFAULT_CONF_THRESHOLD,
    ClassificationResult,
    ProductClassifier,
)

__all__ = [
    # class map
    "RPC_CLASS_NAMES",
    "RPC_CATEGORY_MAP",
    "RPC_CATEGORIES",
    "get_category",
    # classifier
    "ClassificationResult",
    "ProductClassifier",
    "DEFAULT_CONF_THRESHOLD",
]
