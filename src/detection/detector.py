# ============================================================
# Module  : Object Detection & Segmentation
# Owner   : AHAMED A.A.
# Course  : EC9570 – Digital Image Processing
# Desc    : Wraps a YOLOv8 model to detect and localise
#           supermarket products in preprocessed images.
# ============================================================

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

DEFAULT_CONF_THRESHOLD: float = 0.25   # minimum confidence to keep a detection
DEFAULT_IOU_THRESHOLD:  float = 0.45   # NMS IoU threshold (YOLO default)

# BGR colour palette (one per class, cycles if more classes than colours)
_DEFAULT_PALETTE: list[tuple[int, int, int]] = [
    (  0, 200, 255),  # amber-yellow
    ( 50, 205,  50),  # lime-green
    (255,  80,  80),  # coral-blue (BGR -> looks coral on screen)
    (255,   0, 180),  # magenta
    (  0, 165, 255),  # orange
    (147,  20, 255),  # purple
    (  0, 255, 200),  # spring-green
    (255, 191,   0),  # deep sky-blue
]


# ──────────────────────────────────────────────────────────────
# Structured detection result
# ──────────────────────────────────────────────────────────────

class Detection:
    """
    Lightweight container for a single object detection result.

    Attributes
    ----------
    class_id : int
        Integer class index returned by YOLO.
    class_name : str
        Human-readable label (looked-up from *class_names*).
    confidence : float
        Model confidence score in [0, 1].
    bbox : tuple[int, int, int, int]
        Bounding box as ``(x1, y1, x2, y2)`` in pixel coordinates.
    """

    __slots__ = ("class_id", "class_name", "confidence", "bbox")

    def __init__(
        self,
        class_id:   int,
        class_name: str,
        confidence: float,
        bbox:       tuple[int, int, int, int],
    ) -> None:
        self.class_id   = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox       = bbox   # (x1, y1, x2, y2)

    # ── convenience helpers ───────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-friendly)."""
        return {
            "class_id":   self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
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
            f"Detection(class='{self.class_name}', conf={self.confidence:.2f}, "
            f"bbox=({x1},{y1},{x2},{y2}))"
        )


# ──────────────────────────────────────────────────────────────
# ProductDetector
# ──────────────────────────────────────────────────────────────

class ProductDetector:
    """
    Wraps a YOLOv8 model for supermarket product detection.

    Typical usage
    -------------
    >>> detector = ProductDetector()
    >>> detector.load_model("weights/best.pt")
    >>> results = detector.detect(image)
    >>> detections = detector.extract_detections(results, class_names)
    >>> annotated = detector.draw_bounding_boxes(image, detections)
    """

    def __init__(self) -> None:
        self._model: Any | None = None   # ultralytics YOLO instance
        self._model_path: Path | None = None

    # ── 1. load_model ─────────────────────────────────────────

    def load_model(self, model_path: str | Path) -> None:
        """
        Load a YOLOv8 model from *model_path* (typically ``best.pt``).

        Parameters
        ----------
        model_path : str | Path
            Path to the ``.pt`` weights file.

        Raises
        ------
        FileNotFoundError
            If *model_path* does not exist.
        ImportError
            If the ``ultralytics`` package is not installed.
        """
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for ProductDetector. "
                "Install it with:  pip install ultralytics"
            ) from exc

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Model weights not found: {path.resolve()}"
            )

        logger.info("Loading YOLOv8 model from '%s' ...", path)
        self._model = YOLO(str(path))
        self._model_path = path
        logger.info("Model loaded successfully  (%s)", path.name)

    # ── 2. detect ─────────────────────────────────────────────

    def detect(
        self,
        image:          np.ndarray,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold:  float = DEFAULT_IOU_THRESHOLD,
    ) -> Any:
        """
        Run YOLOv8 inference on *image* and return the raw results object.

        Parameters
        ----------
        image : np.ndarray
            BGR image (H x W x 3, dtype=uint8) -- e.g. the output of
            ``preprocess_pipeline()``.
        conf_threshold : float, optional
            Minimum confidence score to retain a detection. Defaults to
            ``DEFAULT_CONF_THRESHOLD`` (0.25).
        iou_threshold : float, optional
            IoU threshold used for Non-Maximum Suppression. Defaults to
            ``DEFAULT_IOU_THRESHOLD`` (0.45).

        Returns
        -------
        ultralytics.engine.results.Results
            Raw YOLO results object.  Pass to :meth:`extract_detections`
            to obtain structured :class:`Detection` instances.

        Raises
        ------
        RuntimeError
            If :meth:`load_model` has not been called yet.
        """
        if self._model is None:
            raise RuntimeError(
                "Model is not loaded. Call load_model(model_path) first."
            )

        logger.debug(
            "Running inference  conf=%.2f  iou=%.2f  image=%s",
            conf_threshold, iou_threshold, image.shape,
        )

        results = self._model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )
        logger.debug("Inference done -- %d result frame(s) returned.", len(results))
        return results

    # ── 3. extract_detections ─────────────────────────────────

    def extract_detections(
        self,
        results:     Any,
        class_names: list[str],
    ) -> list[Detection]:
        """
        Parse raw YOLO *results* into a structured list of :class:`Detection`
        objects.

        Parameters
        ----------
        results : ultralytics Results
            Return value of :meth:`detect`.
        class_names : list[str]
            Ordered list of class label strings whose indices match YOLO's
            class IDs (e.g. ``model.names`` or a custom list).

        Returns
        -------
        list[Detection]
            One :class:`Detection` per bounding box, sorted by confidence
            (highest first).
        """
        detections: list[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                # xyxy tensor -> Python ints
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())

                class_id   = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                # guard against out-of-range class IDs
                if class_id < len(class_names):
                    class_name = class_names[class_id]
                else:
                    class_name = f"class_{class_id}"
                    logger.warning(
                        "class_id %d is out of range (class_names has %d entries).",
                        class_id, len(class_names),
                    )

                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                    )
                )

        # sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)

        logger.info(
            "extract_detections: %d detection(s) parsed.", len(detections)
        )
        return detections

    # ── 4. draw_bounding_boxes ────────────────────────────────

    def draw_bounding_boxes(
        self,
        image:      np.ndarray,
        detections: list[Detection],
        colors:     list[tuple[int, int, int]] | None = None,
    ) -> np.ndarray:
        """
        Annotate *image* with coloured, labelled bounding boxes.

        Parameters
        ----------
        image : np.ndarray
            BGR image to annotate.  The original array is **not** mutated;
            a copy is returned.
        detections : list[Detection]
            Detections produced by :meth:`extract_detections`.
        colors : list[tuple[int, int, int]] | None, optional
            Per-class BGR colour list.  Index is taken modulo ``len(colors)``
            so the palette cycles safely.  Defaults to the built-in
            ``_DEFAULT_PALETTE``.

        Returns
        -------
        np.ndarray
            Annotated BGR image (same shape and dtype as *image*).
        """
        if colors is None:
            colors = _DEFAULT_PALETTE

        annotated = image.copy()
        h, w = annotated.shape[:2]

        # scale text / line thickness relative to image size
        scale      = max(w, h) / 1000.0
        thickness  = max(1, int(2 * scale))
        font_scale = max(0.4, 0.6 * scale)

        for det in detections:
            color = colors[det.class_id % len(colors)]
            x1, y1, x2, y2 = det.bbox

            # ── bounding box ─────────────────────────────────
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # ── label background ─────────────────────────────
            label      = f"{det.class_name}  {det.confidence:.0%}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            label_y1 = max(y1 - th - baseline - 4, 0)
            label_y2 = label_y1 + th + baseline + 4

            cv2.rectangle(
                annotated,
                (x1, label_y1),
                (x1 + tw + 6, label_y2),
                color,
                cv2.FILLED,
            )

            # ── label text (black for contrast) ──────────────
            cv2.putText(
                annotated,
                label,
                (x1 + 3, label_y2 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),   # black text
                thickness,
                cv2.LINE_AA,
            )

        logger.debug(
            "draw_bounding_boxes: annotated %d detection(s) onto %s image.",
            len(detections), annotated.shape,
        )
        return annotated

    # ── repr ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        loaded = str(self._model_path) if self._model_path else "<no model loaded>"
        return f"ProductDetector(model={loaded!r})"
