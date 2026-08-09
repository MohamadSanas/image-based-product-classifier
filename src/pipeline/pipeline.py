"""
pipeline/pipeline.py — End-to-end classification pipeline orchestrator.

Chains: load image → preprocess → segment → classify (per ROI) → aggregate.
Returns a ScanResult domain object.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from src.analytics.statistics import StatisticsEngine
from src.core.entities import DetectedProduct, ScanResult
from src.core.interfaces import IClassifier, IDetector, IPreprocessor

logger = logging.getLogger(__name__)


class ClassificationPipeline:
    """
    Orchestrates the full image processing pipeline.

    Usage
    -----
    >>> pipeline = ClassificationPipeline(preprocessor, segmentor, classifier)
    >>> result: ScanResult = pipeline.run("data/test/shelf.jpg")
    """

    def __init__(
        self,
        preprocessor: IPreprocessor,
        segmentor: IDetector,
        classifier: IClassifier,
        confidence_threshold: float = 0.6,
    ) -> None:
        self.preprocessor = preprocessor
        self.segmentor = segmentor
        self.classifier = classifier
        self.confidence_threshold = confidence_threshold
        self._stats = StatisticsEngine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, image_path: str) -> ScanResult:
        """
        Run the full pipeline on a single image file.

        Parameters
        ----------
        image_path : str
            Path to the input image.

        Returns
        -------
        ScanResult — fully populated domain result object.
        """
        start = time.perf_counter()

        raw_image = self._load_image(image_path)
        preprocessed = self.preprocessor.preprocess(raw_image)
        regions = self.segmentor.detect(preprocessed)

        logger.info("Detected %d product regions in '%s'", len(regions), image_path)

        detected: list[DetectedProduct] = []
        for roi, bbox in regions:
            predictions = self.classifier.predict(roi)
            if not predictions:
                continue

            top_label, top_conf = predictions[0]
            if top_conf < self.confidence_threshold:
                logger.debug(
                    "Region skipped — confidence %.2f < threshold %.2f",
                    top_conf, self.confidence_threshold,
                )
                continue

            detected.append(DetectedProduct(
                label=top_label,
                confidence=top_conf,
                bounding_box=bbox,
                top_k_predictions=predictions,
            ))

        category_counts = self._stats.compute_category_counts(detected)
        duration_ms = (time.perf_counter() - start) * 1000

        result = ScanResult(
            image_path=str(image_path),
            detected_products=detected,
            category_counts=category_counts,
            total_products=len(detected),
            scan_duration_ms=duration_ms,
        )

        logger.info(
            "Pipeline complete: %d products, %.1f ms",
            result.total_products, duration_ms,
        )
        return result

    def run_from_array(self, image: np.ndarray, source_label: str = "array") -> ScanResult:
        """Run the pipeline on a pre-loaded image array (e.g. from API upload)."""
        tmp_path = f"<{source_label}>"
        start = time.perf_counter()

        preprocessed = self.preprocessor.preprocess(image)
        regions = self.segmentor.detect(preprocessed)

        detected: list[DetectedProduct] = []
        for roi, bbox in regions:
            predictions = self.classifier.predict(roi)
            if not predictions:
                continue
            top_label, top_conf = predictions[0]
            if top_conf < self.confidence_threshold:
                continue
            detected.append(DetectedProduct(
                label=top_label,
                confidence=top_conf,
                bounding_box=bbox,
                top_k_predictions=predictions,
            ))

        category_counts = self._stats.compute_category_counts(detected)
        duration_ms = (time.perf_counter() - start) * 1000

        return ScanResult(
            image_path=tmp_path,
            detected_products=detected,
            category_counts=category_counts,
            total_products=len(detected),
            scan_duration_ms=duration_ms,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(image_path: str) -> np.ndarray:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"OpenCV could not read image: {path}")
        return image
