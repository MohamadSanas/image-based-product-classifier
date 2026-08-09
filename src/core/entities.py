"""
core/entities.py — Domain data classes (pure Python, no framework deps).

These are the canonical data structures shared across all layers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x: int        # Top-left x
    y: int        # Top-left y
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return (x, y, x2, y2) for drawing."""
        return (self.x, self.y, self.x2, self.y2)

    def as_xywh(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


@dataclass
class DetectedProduct:
    """A single detected & classified product region in an image."""

    label: str                    # Category name (e.g. "apple")
    confidence: float             # Prediction confidence [0, 1]
    bounding_box: BoundingBox
    top_k_predictions: list[tuple[str, float]] = field(default_factory=list)
    # top_k_predictions: [(label, confidence), ...]


@dataclass
class CategoryCount:
    """Aggregated count for one product category."""

    label: str
    count: int
    percentage: float


@dataclass
class ScanResult:
    """Full result of scanning a single input image."""

    image_path: str
    detected_products: list[DetectedProduct]
    category_counts: list[CategoryCount]
    total_products: int
    scan_timestamp: datetime = field(default_factory=datetime.utcnow)
    scan_duration_ms: Optional[float] = None
    id: Optional[int] = None          # Set after persisting to DB

    @property
    def category_summary(self) -> dict[str, int]:
        """Quick {label: count} mapping."""
        return {c.label: c.count for c in self.category_counts}
