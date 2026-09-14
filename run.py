# ============================================================
# run.py  —  Smart Supermarket Product Identification System
# Owner   : SANAS M.M. & AHAMED A.A.
# Course  : EC9570 – Digital Image Processing
# Desc    : End-to-end pipeline entry point.
#
# Usage
# -----
#   python run.py --image images/20180824-13-50-07-6.jpg
#   python run.py --image images/shelf.jpg --conf 0.35
#   python run.py --help
#
# Pipeline
# --------
#   Input image
#       ↓  preprocess_pipeline()
#   Preprocessed image
#       ↓  ProductDetector.detect() + extract_detections()
#   list[Detection]
#       ↓  ProductClassifier.classify_detections()
#   list[ClassificationResult]
#       ↓  count_by_category()  +  summary()
#   Report  +  Annotated image  →  output/
# ============================================================

from __future__ import annotations

import argparse
import logging
import logging.config
import sys
from pathlib import Path

import cv2
import yaml

# ──────────────────────────────────────────────────────────────
# Logging setup  (load config/logging.yaml if available)
# ──────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent
_LOG_CONFIG   = _PROJECT_ROOT / "config" / "logging.yaml"

if _LOG_CONFIG.is_file():
    with _LOG_CONFIG.open() as _f:
        logging.config.dictConfig(yaml.safe_load(_f))
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

logger = logging.getLogger("pipeline")

# ──────────────────────────────────────────────────────────────
# Add src/ to path so modules resolve without installing
# ──────────────────────────────────────────────────────────────

sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from config_loader import load_config                       # noqa: E402
from detection import ProductDetector                       # noqa: E402
from preprocessing import preprocess_pipeline               # noqa: E402
from classification import ProductClassifier                # noqa: E402
from reporting import ProductVisualizer                     # noqa: E402
from reporting import compute_stats, format_stats_report    # noqa: E402


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Smart Supermarket Product Identification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run.py --image images/shelf.jpg\n"
            "  python run.py --image images/shelf.jpg --conf 0.35\n"
            "  python run.py --image images/shelf.jpg --model models/weights/best.pt\n"
        ),
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        metavar="PATH",
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        metavar="PATH",
        help="Path to YOLOv8 .pt weights file. "
             "Defaults to models/weights/best.pt (from config.yaml).",
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help="Confidence threshold (0–1). "
             "Overrides config.yaml inference.confidence_threshold.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        metavar="DIR",
        help="Root output directory. Defaults to output/ (from config.yaml).",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to config.yaml. Defaults to config/config.yaml.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────

def _save_annotated_image(
    image,
    stem: str,
    output_dir: Path,
) -> Path:
    """Save the annotated (bounding-box) image and return its path."""
    out_dir = output_dir / "detected_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_detected.jpg"
    cv2.imwrite(str(out_path), image)
    return out_path


def _save_report(
    report_text: str,
    stem: str,
    output_dir: Path,
) -> Path:
    """Save the text report and return its path."""
    out_dir = output_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_report.txt"
    out_path.write_text(report_text, encoding="utf-8")
    return out_path


def _save_charts(
    classified: list,
    stem: str,
    output_dir: Path,
) -> list[Path]:
    """Generate and save bar chart, pie chart, and confidence histogram.

    Returns a list of saved chart paths (only charts that were
    successfully created are included).
    """
    viz = ProductVisualizer(output_dir=output_dir / "charts")
    saved: list[Path] = []

    for method_name, label in [
        ("bar_chart",                "Bar chart"),
        ("pie_chart",                "Pie chart"),
        ("confidence_distribution",  "Confidence histogram"),
    ]:
        try:
            method = getattr(viz, method_name)
            path   = method(classified, stem=stem)
            saved.append(path)
            print(f"  [OK] {label:<22} --> {path}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s generation failed: %s", label, exc)

    return saved


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    """
    Execute the full detection-and-classification pipeline.

    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """

    # ── 1. Load config ────────────────────────────────────────
    logger.info("Loading configuration …")
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ImportError) as exc:
        logger.error("Config error: %s", exc)
        return 1

    # ── 2. Resolve paths ─────────────────────────────────────
    image_path  = Path(args.image)
    model_path  = Path(args.model) if args.model else (
        _PROJECT_ROOT / cfg.paths.model_weights
        if hasattr(cfg.paths, "model_weights")
        else _PROJECT_ROOT / "models" / "weights" / "best.pt"
    )
    output_dir  = Path(args.output_dir) if args.output_dir else _PROJECT_ROOT / "output"

    # Override confidence threshold if provided on CLI
    if args.conf is not None:
        cfg.inference.confidence_threshold = args.conf

    logger.info("Image      : %s", image_path)
    logger.info("Model      : %s", model_path)
    logger.info("Output dir : %s", output_dir)
    logger.info("Conf thresh: %.2f", cfg.inference.confidence_threshold)

    # ── 3. Preprocess ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1 — Image Preprocessing")
    print("=" * 60)

    try:
        preprocessed = preprocess_pipeline(image_path, cfg)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Preprocessing failed: %s", exc)
        return 1

    logger.info(
        "Preprocessing complete: original=%s  preprocessed=%s",
        preprocessed.original.shape,
        preprocessed.preprocessed.shape,
    )
    print(f"  [OK] Loaded and preprocessed: {image_path.name}")
    print(f"     Original size : {preprocessed.original.shape[1]}x{preprocessed.original.shape[0]}")
    print(f"     Model input   : {preprocessed.preprocessed.shape[1]}x{preprocessed.preprocessed.shape[0]}")

    # ── 4. Detect ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2 — Object Detection  (YOLOv8)")
    print("=" * 60)

    detector = ProductDetector()
    try:
        detector.load_model(model_path)
    except (FileNotFoundError, ImportError) as exc:
        logger.error("Model loading failed: %s", exc)
        return 1

    classifier = ProductClassifier(cfg)

    raw_results = detector.detect(
        preprocessed.preprocessed,
        conf_threshold=cfg.inference.confidence_threshold,
    )
    detections = detector.extract_detections(raw_results, classifier.class_names)

    print(f"  [OK] Detected {len(detections)} object(s) above threshold {cfg.inference.confidence_threshold:.2f}")

    if not detections:
        print("\n  [!]  No products detected. Try lowering --conf threshold.")
        return 0

    for i, det in enumerate(detections[:5], 1):
        print(f"     {i}. {det.class_name:<30}  conf={det.confidence:.2%}")
    if len(detections) > 5:
        print(f"     … and {len(detections) - 5} more")

    # ── 5. Classify ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3 — Product Classification")
    print("=" * 60)

    classified = classifier.classify_detections(detections)

    print(f"  [OK] Classified {len(classified)} product(s)")

    # ── 6. Count ──────────────────────────────────────────────
    counts = classifier.count_by_category(classified)
    total  = sum(counts.values())

    # ── 7. Console report ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4 — Results")
    print("=" * 60)
    print()
    print(classifier.summary(classified))

    # ── 8. Save annotated image ───────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 5 — Saving Outputs")
    print("=" * 60)

    annotated = detector.draw_bounding_boxes(preprocessed.original, detections)
    stem = image_path.stem
    img_out  = _save_annotated_image(annotated, stem, output_dir)
    print(f"  [OK] Annotated image --> {img_out}")

    # ── 9. Save text report ───────────────────────────────────
    report_lines = [
        f"Smart Supermarket Product Identification System",
        f"EC9570 – Digital Image Processing",
        f"",
        f"Image   : {image_path.name}",
        f"Model   : {model_path.name}",
        f"Threshold: {cfg.inference.confidence_threshold:.2f}",
        f"",
        classifier.summary(classified),
        f"",
        f"Detailed Detections",
        "-" * 60,
    ]
    for r in classified:
        x1, y1, x2, y2 = r.bbox
        report_lines.append(
            f"{r.product_name:<35}  "
            f"cat={r.category:<30}  "
            f"conf={r.confidence:.4f}  "
            f"bbox=({x1},{y1},{x2},{y2})"
        )

    report_text = "\n".join(report_lines)
    txt_out = _save_report(report_text, stem, output_dir)
    print(f"  [OK] Text report     --> {txt_out}")

    # ── 10. Statistical analysis ─────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 6 — Statistical Analysis")
    print("=" * 60)

    stats      = compute_stats(classified)
    stats_text = format_stats_report(stats, image_name=image_path.name)
    print(stats_text)

    stats_out = _save_report(stats_text, f"{stem}_stats", output_dir)
    print(f"  [OK] Stats report    --> {stats_out}")

    # ── 11. Charts ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 7 — Generating Charts")
    print("=" * 60)

    _save_charts(classified, stem=stem, output_dir=output_dir)

    print()
    print("=" * 60)
    print(f"  Pipeline complete.  Total products detected: {total}")
    print("=" * 60)
    print()

    return 0


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(run(_parse_args()))
