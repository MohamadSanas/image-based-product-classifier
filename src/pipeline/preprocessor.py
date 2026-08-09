"""
pipeline/preprocessor.py — Image preprocessing stage.

Implements IPreprocessor. Applies resizing, normalisation, denoising, and
optional CLAHE histogram equalisation to prepare raw images for the pipeline.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from src.core.interfaces import IPreprocessor

logger = logging.getLogger(__name__)


class Preprocessor(IPreprocessor):
    """
    Preprocessing pipeline stage.

    Parameters
    ----------
    target_size : tuple[int, int]
        (height, width) to resize images to. Default (224, 224).
    normalize : bool
        Scale pixel values to [0, 1] if True.
    denoise : bool
        Apply Gaussian blur before processing.
    denoise_kernel : tuple[int, int]
        Kernel size for Gaussian blur.
    apply_clahe : bool
        Apply Contrast Limited Adaptive Histogram Equalisation (per channel).
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (224, 224),
        normalize: bool = True,
        denoise: bool = True,
        denoise_kernel: tuple[int, int] = (3, 3),
        apply_clahe: bool = False,
    ) -> None:
        self.target_size = target_size
        self.normalize = normalize
        self.denoise = denoise
        self.denoise_kernel = denoise_kernel
        self.apply_clahe = apply_clahe
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ------------------------------------------------------------------
    # IPreprocessor implementation
    # ------------------------------------------------------------------

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline.

        Accepts BGR uint8 image, returns float32 image in [0,1] (or uint8
        if normalize=False), resized to target_size.
        """
        if image is None or image.size == 0:
            raise ValueError("Received empty or None image for preprocessing.")

        logger.debug("Preprocessing image with shape %s", image.shape)

        img = self._resize(image)

        if self.denoise:
            img = self._apply_denoise(img)

        if self.apply_clahe:
            img = self._apply_clahe_eq(img)

        if self.normalize:
            img = img.astype(np.float32) / 255.0

        logger.debug("Preprocessed image shape: %s, dtype: %s", img.shape, img.dtype)
        return img

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resize(self, image: np.ndarray) -> np.ndarray:
        h, w = self.target_size
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

    def _apply_denoise(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, self.denoise_kernel, sigmaX=0)

    def _apply_clahe_eq(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE independently on each BGR channel."""
        channels = cv2.split(image)
        eq_channels = [self._clahe.apply(ch) for ch in channels]
        return cv2.merge(eq_channels)

    @classmethod
    def from_config(cls, cfg: dict) -> "Preprocessor":
        """Factory: build from a config dict (loaded from config.yaml)."""
        pre = cfg.get("preprocessing", {})
        return cls(
            target_size=tuple(pre.get("target_size", [224, 224])),
            normalize=pre.get("normalize", True),
            denoise=pre.get("denoise", True),
            denoise_kernel=tuple(pre.get("denoise_kernel", [3, 3])),
            apply_clahe=pre.get("histogram_equalization", False),
        )
