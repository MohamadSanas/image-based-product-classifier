"""
analytics/statistics.py — Aggregation and statistical summary engine.

Computes category counts, percentages, totals, and descriptive summaries
from a list of DetectedProduct objects.
"""
from __future__ import annotations

import logging
from collections import Counter

from src.core.entities import CategoryCount, DetectedProduct

logger = logging.getLogger(__name__)


class StatisticsEngine:
    """
    Computes statistical summaries over a set of detected products.
    This class is pure Python — no framework dependencies.
    """

    def compute_category_counts(
        self,
        detected: list[DetectedProduct],
    ) -> list[CategoryCount]:
        """
        Aggregate detected products into per-category counts with percentages.

        Parameters
        ----------
        detected : list[DetectedProduct]

        Returns
        -------
        list[CategoryCount] sorted by count descending.
        """
        if not detected:
            return []

        total = len(detected)
        raw_counts = Counter(p.label for p in detected)

        counts = [
            CategoryCount(
                label=label,
                count=count,
                percentage=round(count / total * 100, 2),
            )
            for label, count in raw_counts.most_common()
        ]

        logger.debug("Category counts: %s", {c.label: c.count for c in counts})
        return counts

    def summary_text(self, counts: list[CategoryCount], total: int) -> str:
        """
        Generate a plain-text summary table.

        Example
        -------
        Detected Products
        -----------------
        Apple          :  3  (30.00%)
        Orange         :  2  (20.00%)
        ...
        Total Products : 10
        """
        if not counts:
            return "No products detected."

        width = max(len(c.label) for c in counts) + 2
        lines = ["Detected Products", "-" * 40]
        for c in sorted(counts, key=lambda x: x.count, reverse=True):
            lines.append(f"{c.label:<{width}}: {c.count:>3}  ({c.percentage:.2f}%)")

        lines.append("-" * 40)
        lines.append(f"{'Total Products':<{width}}: {total:>3}")
        return "\n".join(lines)

    def average_confidence(self, detected: list[DetectedProduct]) -> float:
        """Return the mean prediction confidence across all detected products."""
        if not detected:
            return 0.0
        return round(sum(p.confidence for p in detected) / len(detected), 4)

    def confidence_by_category(
        self,
        detected: list[DetectedProduct],
    ) -> dict[str, float]:
        """Return {label: mean_confidence} for each category."""
        from collections import defaultdict

        sums: dict[str, list[float]] = defaultdict(list)
        for p in detected:
            sums[p.label].append(p.confidence)

        return {
            label: round(sum(confs) / len(confs), 4)
            for label, confs in sums.items()
        }
