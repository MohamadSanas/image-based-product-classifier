"""
pipeline/feature_extractor.py — Classical feature extraction (optional fallback).

Provides HOG descriptors and colour histograms for use with scikit-learn
classifiers when the deep learning model is unavailable.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from skimage.feature import hog

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts HOG + colour histogram features from a single ROI crop.

    Primarily used as a fallback when no trained CNN is available, or
    for feature engineering experiments in the notebook.
    """

    def __init__(
        self,
        hog_orientations: int = 9,
        hog_pixels_per_cell: tuple[int, int] = (8, 8),
        hog_cells_per_block: tuple[int, int] = (2, 2),
        hist_bins: int = 32,
    ) -> None:
        self.hog_orientations = hog_orientations
        self.hog_pixels_per_cell = hog_pixels_per_cell
        self.hog_cells_per_block = hog_cells_per_block
        self.hist_bins = hist_bins

    def extract(self, roi: np.ndarray) -> np.ndarray:
        """
        Return a 1-D feature vector for a single ROI.

        Parameters
        ----------
        roi : np.ndarray
            Float32 or uint8 BGR image of any size.

        Returns
        -------
        np.ndarray  — Concatenated [HOG | HSV colour histogram] vector.
        """
        # Ensure uint8 for OpenCV
        if roi.dtype != np.uint8:
            roi_u8 = (roi * 255).astype(np.uint8)
        else:
            roi_u8 = roi.copy()

        roi_resized = cv2.resize(roi_u8, (64, 64))
        gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)

        hog_feats = hog(
            gray,
            orientations=self.hog_orientations,
            pixels_per_cell=self.hog_pixels_per_cell,
            cells_per_block=self.hog_cells_per_block,
            feature_vector=True,
        )

        color_feats = self._color_histogram(roi_resized)
        features = np.concatenate([hog_feats, color_feats])
        logger.debug("Extracted feature vector of length %d", len(features))
        return features

    def _color_histogram(self, image: np.ndarray) -> np.ndarray:
        """Compute normalised HSV colour histogram."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        histograms = []
        for ch in range(3):
            hist = cv2.calcHist([hsv], [ch], None, [self.hist_bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histograms.append(hist)
        return np.concatenate(histograms)
