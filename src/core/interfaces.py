"""
core/interfaces.py — Abstract base classes (ports) that define contracts
between layers. Concrete implementations live in pipeline/ and persistence/.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from src.core.entities import DetectedProduct, ScanResult


class IPreprocessor(ABC):
    """Contract for the image preprocessing stage."""

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Accept a raw BGR image (H×W×3 uint8) and return a preprocessed
        image ready for segmentation or direct classification.
        """


class IDetector(ABC):
    """Contract for the product detection / segmentation stage."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[tuple[np.ndarray, "BoundingBox"]]:  # noqa: F821
        """
        Accept a preprocessed image and return a list of (roi_crop, bbox) pairs —
        one per detected product region.
        """


class IClassifier(ABC):
    """Contract for the classification stage."""

    @abstractmethod
    def predict(self, roi: np.ndarray) -> list[tuple[str, float]]:
        """
        Accept a single ROI crop and return top-k (label, confidence) tuples
        sorted by descending confidence.
        """

    @abstractmethod
    def load(self, weights_path: str) -> None:
        """Load model weights from disk."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """True when model weights have been loaded."""


class IScanRepository(ABC):
    """Contract for persisting and retrieving scan results."""

    @abstractmethod
    def save(self, result: ScanResult) -> ScanResult:
        """Persist a scan result and return it with its assigned id."""

    @abstractmethod
    def get_by_id(self, scan_id: int) -> Optional[ScanResult]:
        """Retrieve a single scan result by primary key."""

    @abstractmethod
    def get_all(self, limit: int = 50, offset: int = 0) -> list[ScanResult]:
        """Retrieve paginated scan history, newest first."""

    @abstractmethod
    def delete(self, scan_id: int) -> bool:
        """Delete a scan record. Returns True if deleted, False if not found."""
