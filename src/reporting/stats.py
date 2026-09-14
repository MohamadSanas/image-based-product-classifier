# ============================================================
# Module  : Statistical Analysis
# Owner   : AHAMED A.A.
# Course  : EC9570 – Digital Image Processing
# Desc    : Computes descriptive statistics (counts, percentages,
#           mean confidence, etc.) from ClassificationResult data
#           and formats them into a human-readable text report.
# ============================================================

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from classification import ClassificationResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────

@dataclass
class CategoryStats:
    """
    Aggregated statistics for a single product category.

    Attributes
    ----------
    name : str
        Category name (one of the 17 RPC categories).
    count : int
        Number of detected products in this category.
    percentage : float
        Share of total detections (0-100).
    mean_confidence : float
        Average YOLO confidence across products in this category.
    max_confidence : float
        Highest single-detection confidence in this category.
    min_confidence : float
        Lowest single-detection confidence in this category.
    products : list[str]
        Fine-grained product names detected within this category.
    """
    name:            str
    count:           int
    percentage:      float
    mean_confidence: float
    max_confidence:  float
    min_confidence:  float
    products:        list[str] = field(default_factory=list)


@dataclass
class DetectionStats:
    """
    Full detection-level statistics for a single run.

    Attributes
    ----------
    total_detections : int
        Total number of accepted detections.
    num_categories : int
        Number of unique major product categories detected.
    num_unique_products : int
        Number of unique fine-grained product names detected.
    mean_confidence : float
        Mean confidence across all detections.
    max_confidence : float
        Highest single-detection confidence.
    min_confidence : float
        Lowest single-detection confidence.
    by_category : dict[str, CategoryStats]
        Per-category stats, ordered by count descending.
    by_product : dict[str, int]
        ``{product_name: count}`` ordered by count descending.
    """
    total_detections:    int
    num_categories:      int
    num_unique_products: int
    mean_confidence:     float
    max_confidence:      float
    min_confidence:      float
    by_category:         dict[str, CategoryStats] = field(default_factory=dict)
    by_product:          dict[str, int]            = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def compute_stats(results: list) -> DetectionStats:
    """
    Compute descriptive statistics from a list of classification results.

    Parameters
    ----------
    results : list[ClassificationResult]
        Output of ``ProductClassifier.classify_detections()``.

    Returns
    -------
    DetectionStats
        Fully populated statistics object.
    """
    if not results:
        logger.warning("compute_stats called with empty results list.")
        return DetectionStats(
            total_detections=0,
            num_categories=0,
            num_unique_products=0,
            mean_confidence=0.0,
            max_confidence=0.0,
            min_confidence=0.0,
        )

    total = len(results)
    confidences = [r.confidence for r in results]

    # ── product-level counts ───────────────────────────────────
    product_counter: Counter[str] = Counter(r.product_name for r in results)
    by_product = dict(
        sorted(product_counter.items(), key=lambda kv: (-kv[1], kv[0]))
    )

    # ── per-category aggregation ───────────────────────────────
    category_groups: dict[str, list] = {}
    for r in results:
        category_groups.setdefault(r.category, []).append(r)

    by_category: dict[str, CategoryStats] = {}
    for cat_name, group in sorted(
        category_groups.items(), key=lambda kv: -len(kv[1])
    ):
        confs = [r.confidence for r in group]
        by_category[cat_name] = CategoryStats(
            name=cat_name,
            count=len(group),
            percentage=round(len(group) / total * 100, 2),
            mean_confidence=round(sum(confs) / len(confs), 4),
            max_confidence=round(max(confs), 4),
            min_confidence=round(min(confs), 4),
            products=sorted({r.product_name for r in group}),
        )

    stats = DetectionStats(
        total_detections=total,
        num_categories=len(by_category),
        num_unique_products=len(by_product),
        mean_confidence=round(sum(confidences) / total, 4),
        max_confidence=round(max(confidences), 4),
        min_confidence=round(min(confidences), 4),
        by_category=by_category,
        by_product=by_product,
    )

    logger.info(
        "compute_stats: %d detections, %d categories, %d unique products.",
        total, stats.num_categories, stats.num_unique_products,
    )
    return stats


def format_stats_report(stats: DetectionStats, image_name: str = "") -> str:
    """
    Format a ``DetectionStats`` object as a structured plain-text report.

    Parameters
    ----------
    stats : DetectionStats
        Output of :func:`compute_stats`.
    image_name : str, optional
        Name of the source image to embed in the report header.

    Returns
    -------
    str
        Multi-line report string ready to print or write to a file.
    """
    W = 60
    SEP = "=" * W
    sep = "-" * W

    lines: list[str] = [
        SEP,
        "  STATISTICAL ANALYSIS REPORT",
        "  Smart Supermarket Product Identification System",
        "  EC9570 - Digital Image Processing",
        SEP,
        "",
    ]

    if image_name:
        lines.append(f"  Image              : {image_name}")

    if stats.total_detections == 0:
        lines += ["", "  [!] No products detected.", ""]
        return "\n".join(lines)

    lines += [
        f"  Total Detections   : {stats.total_detections}",
        f"  Categories Found   : {stats.num_categories}",
        f"  Unique Products    : {stats.num_unique_products}",
        f"  Mean Confidence    : {stats.mean_confidence:.1%}",
        f"  Max  Confidence    : {stats.max_confidence:.1%}",
        f"  Min  Confidence    : {stats.min_confidence:.1%}",
        "",
        sep,
        "  BY CATEGORY",
        sep,
        "",
    ]

    lines.append(f"  {'Category':<32} {'Count':>5}  {'Share':>6}  {'AvgConf':>7}")
    lines.append(f"  {'-'*32} {'-'*5}  {'-'*6}  {'-'*7}")

    for cat_name, cs in stats.by_category.items():
        bar_len = max(1, round(cs.percentage / 5))
        bar = "#" * bar_len
        lines.append(
            f"  {cat_name:<32} {cs.count:>5}  {cs.percentage:>5.1f}%"
            f"  {cs.mean_confidence:>6.1%}   {bar}"
        )

    lines += [
        "",
        sep,
        "  BY PRODUCT  (top 15)",
        sep,
        "",
    ]

    top_products = list(stats.by_product.items())[:15]
    for rank, (prod_name, count) in enumerate(top_products, 1):
        lines.append(f"  {rank:>2}. {prod_name:<38}  x{count}")

    lines += ["", SEP, ""]
    return "\n".join(lines)
