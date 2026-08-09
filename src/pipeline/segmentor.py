"""
pipeline/segmentor.py — Product region detection / segmentation stage.

Implements IDetector. Supports three strategies:
  - contour  : thresholding + morphological ops + contour finding (fast)
  - grabcut  : GrabCut iterative segmentation (accurate, slower)
  - watershed: Watershed algorithm (good for touching objects)

Returns a list of (roi_crop, BoundingBox) pairs.
"""
from __future__ import annotations

import logging
from typing import Literal

import cv2
import numpy as np

from src.core.entities import BoundingBox
from src.core.interfaces import IDetector

logger = logging.getLogger(__name__)

SegmentationMethod = Literal["contour", "grabcut", "watershed"]


class Segmentor(IDetector):
    """
    Multi-strategy product region segmentor.

    Parameters
    ----------
    method : SegmentationMethod
        Algorithm to use for detecting product regions.
    min_area : int
        Minimum contour/region area in pixels² to keep.
    padding : int
        Pixels of padding to add around each detected bounding box.
    dilation_kernel : tuple[int, int]
        Kernel size for morphological dilation (contour method).
    """

    def __init__(
        self,
        method: SegmentationMethod = "contour",
        min_area: int = 500,
        padding: int = 10,
        dilation_kernel: tuple[int, int] = (5, 5),
    ) -> None:
        self.method = method
        self.min_area = min_area
        self.padding = padding
        self.dilation_kernel = dilation_kernel

    # ------------------------------------------------------------------
    # IDetector implementation
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> list[tuple[np.ndarray, BoundingBox]]:
        """
        Detect product regions in a preprocessed image.

        Parameters
        ----------
        image : np.ndarray
            Float32 [0,1] or uint8 BGR image.

        Returns
        -------
        list of (roi_crop, BoundingBox) — one per detected region.
        """
        # Ensure uint8 for OpenCV operations
        if image.dtype != np.uint8:
            img_u8 = (image * 255).astype(np.uint8)
        else:
            img_u8 = image.copy()

        logger.debug("Running segmentation [method=%s] on %s", self.method, img_u8.shape)

        if self.method == "contour":
            regions = self._detect_contour(img_u8)
        elif self.method == "grabcut":
            regions = self._detect_grabcut(img_u8)
        elif self.method == "watershed":
            regions = self._detect_watershed(img_u8)
        else:
            raise ValueError(f"Unknown segmentation method: {self.method}")

        logger.info("Segmentor found %d product regions.", len(regions))
        return regions

    # ------------------------------------------------------------------
    # Contour-based detection
    # ------------------------------------------------------------------

    def _detect_contour(self, image: np.ndarray) -> list[tuple[np.ndarray, BoundingBox]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold for varying illumination
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11, C=2,
        )

        # Morphological operations to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.dilation_kernel)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        dilated = cv2.dilate(closed, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self._contours_to_regions(image, contours)

    def _contours_to_regions(
        self, image: np.ndarray, contours: list
    ) -> list[tuple[np.ndarray, BoundingBox]]:
        h, w = image.shape[:2]
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)

            # Apply padding with boundary clipping
            x1 = max(0, x - self.padding)
            y1 = max(0, y - self.padding)
            x2 = min(w, x + bw + self.padding)
            y2 = min(h, y + bh + self.padding)

            roi = image[y1:y2, x1:x2]
            bbox = BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
            regions.append((roi, bbox))

        return regions

    # ------------------------------------------------------------------
    # GrabCut-based detection
    # ------------------------------------------------------------------

    def _detect_grabcut(self, image: np.ndarray) -> list[tuple[np.ndarray, BoundingBox]]:
        """GrabCut applied with a central rectangle as initial foreground rect."""
        h, w = image.shape[:2]
        margin_x, margin_y = w // 8, h // 8
        rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

        mask = np.zeros((h, w), np.uint8)
        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(image, mask, rect, bg_model, fg_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

        # Pixels labelled 2 (probable bg) or 0 (bg) → 0, rest → 1
        fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
        fg_image = image * fg_mask[:, :, np.newaxis]

        # Find contours on the resulting mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self._contours_to_regions(fg_image, contours)

    # ------------------------------------------------------------------
    # Watershed-based detection
    # ------------------------------------------------------------------

    def _detect_watershed(self, image: np.ndarray) -> list[tuple[np.ndarray, BoundingBox]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)

        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers += 1
        markers[unknown == 255] = 0

        markers = cv2.watershed(image, markers)

        regions = []
        for label in np.unique(markers):
            if label in (-1, 1):  # border and background
                continue
            component_mask = np.zeros(gray.shape, dtype=np.uint8)
            component_mask[markers == label] = 255
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                regions.extend(self._contours_to_regions(image, contours))

        return regions

    @classmethod
    def from_config(cls, cfg: dict) -> "Segmentor":
        seg = cfg.get("segmentation", {})
        return cls(
            method=seg.get("method", "contour"),
            min_area=seg.get("min_contour_area", 500),
            padding=seg.get("padding", 10),
            dilation_kernel=tuple(seg.get("dilation_kernel", [5, 5])),
        )
