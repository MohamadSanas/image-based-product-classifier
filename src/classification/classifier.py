# ============================================================
# Module  : Product Classification
# Owner   : SANAS M.M.
# Course  : EC9570 – Digital Image Processing
# Desc    : Maps raw YOLO Detection objects (from the detection
#           module) to structured ClassificationResult objects
#           that carry the fine-grained product name, the major
#           product category, and confidence information.
#           Also provides product counting and top-k helpers.
# ============================================================

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, NamedTuple

from .class_map import RPC_CLASS_NAMES, get_category

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

DEFAULT_CONF_THRESHOLD: float = 0.25
"""Minimum confidence score to accept a detection as a classification."""


# ──────────────────────────────────────────────────────────────
# Structured classification result
# ──────────────────────────────────────────────────────────────

class ClassificationResult(NamedTuple):
    """
    Enriched result for a single detected product.

    Attributes
    ----------
    class_id : int
        Integer YOLO class index.
    product_name : str
        Fine-grained RPC product class name (from ``RPC_CLASS_NAMES``).
    category : str
        Major product category (one of the 17 RPC categories).
    confidence : float
        YOLO detection confidence in [0, 1].
    bbox : tuple[int, int, int, int]
        Bounding box as ``(x1, y1, x2, y2)`` in pixel coordinates.
    """

    class_id:     int
    product_name: str
    category:     str
    confidence:   float
    bbox:         tuple[int, int, int, int]

    # ── helpers ───────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-friendly)."""
        return {
            "class_id":     self.class_id,
            "product_name": self.product_name,
            "category":     self.category,
            "confidence":   round(float(self.confidence), 4),
            "bbox": {
                "x1": self.bbox[0],
                "y1": self.bbox[1],
                "x2": self.bbox[2],
                "y2": self.bbox[3],
            },
        }

    def __repr__(self) -> str:
        x1, y1, x2, y2 = self.bbox
        return (
            f"ClassificationResult("
            f"product='{self.product_name}', "
            f"category='{self.category}', "
            f"conf={self.confidence:.2f}, "
            f"bbox=({x1},{y1},{x2},{y2}))"
        )


# ──────────────────────────────────────────────────────────────
# ProductClassifier
# ──────────────────────────────────────────────────────────────

class ProductClassifier:
    """
    Maps YOLO ``Detection`` objects to rich ``ClassificationResult``
    objects by resolving each class ID against the RPC class map.

    Typical usage
    -------------
    >>> from src.detection import ProductDetector
    >>> from src.classification import ProductClassifier
    >>>
    >>> detector   = ProductDetector()
    >>> detector.load_model("models/weights/best.pt")
    >>> classifier = ProductClassifier()
    >>>
    >>> results_raw  = detector.detect(preprocessed_image)
    >>> detections   = detector.extract_detections(results_raw, classifier.class_names)
    >>> classified   = classifier.classify_detections(detections)
    >>> counts       = classifier.count_by_category(classified)
    >>> top3         = classifier.top_k(classified, k=3)
    """

    def __init__(self, cfg: Any | None = None) -> None:
        """
        Initialise the classifier.

        Parameters
        ----------
        cfg : object | None
            Optional config namespace (from ``config_loader.load_config``).
            Expected attributes::

                cfg.inference.confidence_threshold  – float (default 0.25)

            When *cfg* is ``None``, ``DEFAULT_CONF_THRESHOLD`` is used.
        """
        self._cfg = cfg

        # Resolve confidence threshold from config or use default
        if cfg is not None and hasattr(cfg, "inference"):
            self._conf_threshold: float = float(cfg.inference.confidence_threshold)
        else:
            self._conf_threshold = DEFAULT_CONF_THRESHOLD

        logger.debug(
            "ProductClassifier initialised  conf_threshold=%.2f",
            self._conf_threshold,
        )

    # ── public properties ─────────────────────────────────────

    @property
    def class_names(self) -> list[str]:
        """
        Ordered list of 200 RPC class name strings.

        Pass this to ``ProductDetector.extract_detections()`` so the
        detector and classifier share the same label vocabulary:

        >>> detections = detector.extract_detections(raw, classifier.class_names)
        """
        return RPC_CLASS_NAMES

    @property
    def conf_threshold(self) -> float:
        """Confidence threshold used by :meth:`classify_detections`."""
        return self._conf_threshold

    # ── 1. classify_detections ────────────────────────────────

    def classify_detections(
        self,
        detections: list[Any],
        conf_threshold: float | None = None,
    ) -> list[ClassificationResult]:
        """
        Convert a list of ``Detection`` objects to ``ClassificationResult``
        objects, filtering by confidence and enriching with category info.

        Parameters
        ----------
        detections : list[Detection]
            Output of ``ProductDetector.extract_detections()``.
        conf_threshold : float | None, optional
            Override the instance-level confidence threshold for this call.
            Detections with ``confidence < conf_threshold`` are discarded.
            Defaults to the threshold set in ``__init__``.

        Returns
        -------
        list[ClassificationResult]
            One result per accepted detection, sorted by confidence
            (highest first).
        """
        threshold = conf_threshold if conf_threshold is not None else self._conf_threshold

        results: list[ClassificationResult] = []

        for det in detections:
            if det.confidence < threshold:
                logger.debug(
                    "Skipping '%s' — conf %.2f < threshold %.2f",
                    det.class_name, det.confidence, threshold,
                )
                continue

            category = get_category(det.class_name, default="Unknown")

            results.append(
                ClassificationResult(
                    class_id=det.class_id,
                    product_name=det.class_name,
                    category=category,
                    confidence=det.confidence,
                    bbox=det.bbox,
                )
            )

        # Already sorted by detector; keep highest-confidence first
        results.sort(key=lambda r: r.confidence, reverse=True)

        logger.info(
            "classify_detections: %d/%d detections accepted (threshold=%.2f).",
            len(results), len(detections), threshold,
        )
        return results

    # ── 2. count_by_category ──────────────────────────────────

    def count_by_category(
        self,
        results: list[ClassificationResult],
    ) -> dict[str, int]:
        """
        Count detected products grouped by major product category.

        Parameters
        ----------
        results : list[ClassificationResult]
            Output of :meth:`classify_detections`.

        Returns
        -------
        dict[str, int]
            Mapping ``{category_name: count}``, sorted by count descending.

        Examples
        --------
        >>> counts = classifier.count_by_category(classified)
        >>> # {"Chocolate": 4, "Soft Drinks & Beverages": 3, ...}
        """
        counter: Counter[str] = Counter(r.category for r in results)
        # Sort by count descending, then alphabetically for stable output
        sorted_counts = dict(
            sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        )
        logger.info(
            "count_by_category: %d unique categories from %d detections.",
            len(sorted_counts), len(results),
        )
        return sorted_counts

    # ── 3. count_by_product ───────────────────────────────────

    def count_by_product(
        self,
        results: list[ClassificationResult],
    ) -> dict[str, int]:
        """
        Count detected products grouped by fine-grained product name.

        Parameters
        ----------
        results : list[ClassificationResult]
            Output of :meth:`classify_detections`.

        Returns
        -------
        dict[str, int]
            Mapping ``{product_name: count}``, sorted by count descending.
        """
        counter: Counter[str] = Counter(r.product_name for r in results)
        sorted_counts = dict(
            sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        )
        logger.info(
            "count_by_product: %d unique products from %d detections.",
            len(sorted_counts), len(results),
        )
        return sorted_counts

    # ── 4. top_k ──────────────────────────────────────────────

    def top_k(
        self,
        results: list[ClassificationResult],
        k: int = 3,
    ) -> list[ClassificationResult]:
        """
        Return the top-*k* most confident ``ClassificationResult`` objects.

        Parameters
        ----------
        results : list[ClassificationResult]
            Output of :meth:`classify_detections` (assumed sorted by
            confidence descending; if not, they will be re-sorted here).
        k : int
            Number of results to return.  If *k* > ``len(results)`` all
            results are returned.  Defaults to 3.

        Returns
        -------
        list[ClassificationResult]
            Up to *k* results, highest confidence first.
        """
        if k < 1:
            raise ValueError(f"k must be a positive integer; got {k}.")

        sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)
        top = sorted_results[:k]

        logger.debug("top_k: returning %d of %d results.", len(top), len(results))
        return top

    # ── 5. summary ────────────────────────────────────────────

    def summary(self, results: list[ClassificationResult]) -> str:
        """
        Build a human-readable plain-text summary of classified detections.

        The output matches the "Example Output" format shown in README.md::

            Detected Products
            -----------------
            Chocolate : 4
            Soft Drinks & Beverages : 3
            Milk & Dairy Products : 2

            Total Products : 9

        Parameters
        ----------
        results : list[ClassificationResult]
            Output of :meth:`classify_detections`.

        Returns
        -------
        str
            Multi-line summary string, ready to print or write to a file.
        """
        counts = self.count_by_category(results)
        total  = sum(counts.values())

        lines: list[str] = [
            "Detected Products",
            "-" * 40,
        ]

        for category, count in counts.items():
            lines.append(f"{category:<35}: {count}")

        lines += [
            "-" * 40,
            f"Total Products : {total}",
        ]

        if total > 0:
            lines += [
                "",
                "Distribution",
                "-" * 40,
            ]
            for category, count in counts.items():
                pct = count / total * 100
                lines.append(f"{category:<35}  {pct:.1f}%")

        return "\n".join(lines)

    # ── repr ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"ProductClassifier("
            f"conf_threshold={self._conf_threshold:.2f}, "
            f"num_classes={len(RPC_CLASS_NAMES)})"
        )
