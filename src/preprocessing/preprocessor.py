# ============================================================
# Module  : Image Acquisition & Preprocessing
# Owner   : SANAS M.M.
# Course  : EC9570 – Digital Image Processing
# Desc    : Loads, validates, and preprocesses supermarket
#           product images before feeding into the detector.
# ============================================================

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_TARGET_SIZE: tuple[int, int] = (224, 224)   # (height, width)
YOLO_TARGET_SIZE:    tuple[int, int] = (640, 640)   # (height, width)


# ──────────────────────────────────────────────────────────────
# Return type for the pipeline
# ──────────────────────────────────────────────────────────────

class PreprocessedImage(NamedTuple):
    """Holds both the original and the fully preprocessed image."""
    original:     np.ndarray   # BGR, unmodified
    preprocessed: np.ndarray   # BGR, ready for the detector
    path:         Path          # source file path


# ──────────────────────────────────────────────────────────────
# 1. Image Loading & Validation
# ──────────────────────────────────────────────────────────────

def load_image(path: str | Path) -> np.ndarray:
    """
    Read an image from *path* using OpenCV and validate it.

    Parameters
    ----------
    path : str | Path
        Absolute or relative path to the image file.

    Returns
    -------
    np.ndarray
        BGR image array (H × W × 3, dtype=uint8).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not in ``ALLOWED_EXTENSIONS`` or the
        image cannot be decoded by OpenCV.
    """
    path = Path(path)

    # ── Existence check ──────────────────────────────────────
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # ── Extension check ──────────────────────────────────────
    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    # ── Load ─────────────────────────────────────────────────
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"OpenCV could not decode the image: {path}")

    logger.info("Loaded image '%s'  shape=%s  dtype=%s", path.name, img.shape, img.dtype)
    return img


# ──────────────────────────────────────────────────────────────
# 2. Resize
# ──────────────────────────────────────────────────────────────

def resize_image(
    img: np.ndarray,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    *,
    keep_aspect: bool = True,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Resize *img* to *target_size* (height, width).

    Parameters
    ----------
    img : np.ndarray
        BGR image.
    target_size : tuple[int, int]
        Desired (height, width). Defaults to 224 × 224 for MobileNet /
        EfficientNet classifiers; use (640, 640) for YOLO.
    keep_aspect : bool
        If ``True``, letterbox the image so the aspect ratio is
        preserved and the shortfall is padded with black pixels.
        If ``False``, the image is stretched to exactly *target_size*.
    interpolation : int
        OpenCV interpolation flag (default: ``cv2.INTER_LINEAR``).

    Returns
    -------
    np.ndarray
        Resized BGR image of shape (target_height, target_width, 3).
    """
    th, tw = target_size
    h,  w  = img.shape[:2]

    if not keep_aspect:
        resized = cv2.resize(img, (tw, th), interpolation=interpolation)
        logger.debug("Resized %s→%s (stretch)", (h, w), (th, tw))
        return resized

    # ── Letterbox ────────────────────────────────────────────
    scale  = min(tw / w, th / h)
    new_w  = int(w * scale)
    new_h  = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    pad_top  = (th - new_h) // 2
    pad_left = (tw - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    logger.debug("Resized %s→%s (letterbox, scale=%.4f)", (h, w), (th, tw), scale)
    return canvas


# ──────────────────────────────────────────────────────────────
# 3. Denoising
# ──────────────────────────────────────────────────────────────

def denoise_image(
    img: np.ndarray,
    kernel: tuple[int, int] = (3, 3),
    sigma: float = 0,
) -> np.ndarray:
    """
    Apply Gaussian blur to remove high-frequency noise.

    Parameters
    ----------
    img : np.ndarray
        BGR image.
    kernel : tuple[int, int]
        Kernel size (width, height). Both values must be odd and positive.
        Defaults to (3, 3) — gentle smoothing that preserves edges.
    sigma : float
        Standard deviation in X (and Y if sigmaY=0). When 0 OpenCV
        computes it automatically from the kernel size.

    Returns
    -------
    np.ndarray
        Denoised BGR image.
    """
    kw, kh = kernel

    # Kernel dimensions must be odd
    kw = kw if kw % 2 == 1 else kw + 1
    kh = kh if kh % 2 == 1 else kh + 1

    denoised = cv2.GaussianBlur(img, (kw, kh), sigma)
    logger.debug("Denoised with GaussianBlur kernel=(%d,%d)", kw, kh)
    return denoised


# ──────────────────────────────────────────────────────────────
# 4. Normalisation
# ──────────────────────────────────────────────────────────────

def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Scale pixel values from [0, 255] to [0.0, 1.0].

    The output dtype is ``float32`` — compatible with both TensorFlow
    and PyTorch model inputs.

    Parameters
    ----------
    img : np.ndarray
        BGR image with dtype uint8.

    Returns
    -------
    np.ndarray
        Normalised image with dtype float32, same shape as *img*.
    """
    normalised = img.astype(np.float32) / 255.0
    logger.debug("Normalised image to [0, 1]  min=%.4f  max=%.4f",
                 normalised.min(), normalised.max())
    return normalised


# ──────────────────────────────────────────────────────────────
# 5. CLAHE (Contrast Limited Adaptive Histogram Equalisation)
# ──────────────────────────────────────────────────────────────

def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Enhance local contrast using CLAHE on the L channel of LAB colour space.

    Useful when products are photographed under uneven lighting or when
    the image has low-contrast regions.

    Parameters
    ----------
    img : np.ndarray
        BGR image (uint8).
    clip_limit : float
        Threshold for contrast limiting. Higher values give stronger
        enhancement but may amplify noise. Default 2.0.
    tile_grid_size : tuple[int, int]
        Size of the grid for histogram equalization. Default (8, 8).

    Returns
    -------
    np.ndarray
        Contrast-enhanced BGR image (uint8).
    """
    lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe   = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)

    enhanced_lab = cv2.merge([l_clahe, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    logger.debug("Applied CLAHE  clip_limit=%.1f  tile=%s", clip_limit, tile_grid_size)
    return enhanced_bgr


# ──────────────────────────────────────────────────────────────
# 6. Full Preprocessing Pipeline
# ──────────────────────────────────────────────────────────────

def preprocess_pipeline(
    path: str | Path,
    cfg: Any | None = None,
) -> PreprocessedImage:
    """
    Run the complete preprocessing pipeline for a single image.

    Pipeline steps (driven by *cfg*; sensible defaults used when None):

    1. Load & validate image
    2. Resize to detector input size  (640 × 640 for YOLO)
    3. Denoise with Gaussian blur
    4. Optional CLAHE histogram equalisation
    5. Returns **both** the original and the preprocessed image

    .. note::
        Normalisation to [0, 1] is intentionally **not** applied here
        because YOLOv8 handles its own internal normalisation.  If you
        are feeding crops into a standalone MobileNetV2 classifier call
        ``normalize_image()`` on the crop afterwards.

    Parameters
    ----------
    path : str | Path
        Path to the image file.
    cfg : object | None
        Configuration namespace (from ``config_loader.load_config``).
        Expected attributes::

            cfg.preprocessing.target_size         – [height, width]
            cfg.preprocessing.denoise             – bool
            cfg.preprocessing.denoise_kernel      – [kw, kh]
            cfg.preprocessing.histogram_equalization – bool

        When *cfg* is ``None``, default values are used.

    Returns
    -------
    PreprocessedImage
        Named tuple with fields:

        - ``original``     – unmodified BGR image (uint8)
        - ``preprocessed`` – processed BGR image  (uint8, detector-ready)
        - ``path``         – resolved ``Path`` of the source file
    """
    path = Path(path).resolve()

    # ── 1. Load ───────────────────────────────────────────────
    original = load_image(path)
    processed = original.copy()

    # ── 2. Resize (Optional — YOLO handles 640x640 scaling internally) ──
    do_resize = getattr(cfg.preprocessing, "resize", False) if cfg is not None else False
    if do_resize:
        h, w = cfg.preprocessing.target_size if cfg is not None else YOLO_TARGET_SIZE
        processed = resize_image(processed, target_size=(h, w), keep_aspect=True)

    # ── 3. Denoise ────────────────────────────────────────────
    do_denoise = cfg.preprocessing.denoise if cfg is not None else True
    if do_denoise:
        kernel_cfg = cfg.preprocessing.denoise_kernel if cfg is not None else [3, 3]
        kernel: tuple[int, int] = (int(kernel_cfg[0]), int(kernel_cfg[1]))  # list → (w, h)
        processed = denoise_image(processed, kernel=kernel)

    # ── 4. CLAHE (optional) ───────────────────────────────────
    do_clahe = cfg.preprocessing.histogram_equalization if cfg is not None else False
    if do_clahe:
        processed = apply_clahe(processed)

    logger.info(
        "Preprocessing complete for '%s'  "
        "original=%s  preprocessed=%s",
        path.name, original.shape, processed.shape,
    )

    return PreprocessedImage(
        original=original,
        preprocessed=processed,
        path=path,
    )
