# ============================================================
# app.py  —  Smart Supermarket AI Checkout Web Server
# Course  : EC9570 – Digital Image Processing
# Authors : SANAS M.M. & AHAMED A.A.
# Desc    : FastAPI backend serving the web application and
#           live product identification pipeline.
# ============================================================

from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Setup import path for src
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from config_loader import load_config
from detection import ProductDetector
from preprocessing import denoise_image, apply_clahe, resize_image
from classification import ProductClassifier
from reporting import ProductVisualizer, compute_stats, format_stats_report

# ──────────────────────────────────────────────────────────────
# App Initialization & Model Pre-loading
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="SmartMart AI Checkout API",
    description="Digital Image Processing & YOLOv8 Product Identification System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
STATIC_DIR = _PROJECT_ROOT / "static"
IMAGES_DIR = _PROJECT_ROOT / "images"
OUTPUT_DIR = _PROJECT_ROOT / "output"
WEIGHTS_PATH = _PROJECT_ROOT / "models" / "weights" / "best.pt"

STATIC_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Mount static and output assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

# Preload YOLO Detector
print("\n" + "=" * 60)
print("  Initializing SmartMart AI Checkout Pipeline...")
print("=" * 60)

cfg = load_config()
detector = ProductDetector()
if WEIGHTS_PATH.exists():
    detector.load_model(WEIGHTS_PATH)
    print(f"  [OK] Model weights loaded: {WEIGHTS_PATH.name}")
else:
    print(f"  [WARN] Model weights not found at {WEIGHTS_PATH}")

classifier = ProductClassifier(cfg)
visualizer = ProductVisualizer(output_dir=OUTPUT_DIR / "charts")
print("  [OK] Classification & Reporting modules ready.")
print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _image_to_base64(img_bgr: np.ndarray, quality: int = 85) -> str:
    """Encode OpenCV BGR image to base64 JPEG data URL."""
    success, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def _file_to_base64(file_path: Path) -> str:
    """Read a PNG or JPG file and return base64 data URL."""
    if not file_path.exists():
        return ""
    ext = file_path.suffix.lower().replace(".", "")
    mime = "image/png" if ext == "png" else "image/jpeg"
    with open(file_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────

@app.get("/")
@app.head("/")
def serve_index():
    """Serve the single page application."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(str(index_file))


@app.get("/api/samples")
def list_sample_images():
    """List all available supermarket test images."""
    if not IMAGES_DIR.exists():
        return {"samples": []}
    
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    samples = sorted([
        p.name for p in IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    ])
    return {"samples": samples, "total": len(samples)}


@app.get("/api/sample-image/{filename}")
def get_sample_image(filename: str):
    """Serve a sample image for quick previews."""
    img_path = IMAGES_DIR / filename
    if not img_path.exists() or not img_path.is_file():
        raise HTTPException(status_code=404, detail="Sample image not found.")
    return FileResponse(str(img_path))


@app.post("/api/detect")
async def detect_products(
    file: UploadFile | None = File(None),
    sample_name: str | None = Form(None),
    conf: float = Form(0.50),
    denoise: bool = Form(True),
    clahe: bool = Form(False),
):
    """
    Run the complete product detection, isolation, classification,
    and statistical reporting pipeline on the submitted image.
    """
    # 1. Image Acquisition
    img_bytes = None
    source_stem = f"scan_{int(time.time())}"

    if file and file.filename:
        img_bytes = await file.read()
        source_stem = Path(file.filename).stem
    elif sample_name:
        sample_path = IMAGES_DIR / sample_name
        if not sample_path.exists():
            raise HTTPException(status_code=404, detail="Selected sample not found.")
        img_bytes = sample_path.read_bytes()
        source_stem = sample_path.stem
    else:
        raise HTTPException(status_code=400, detail="No image file or sample provided.")

    np_arr = np.frombuffer(img_bytes, np.uint8)
    original = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if original is None:
        raise HTTPException(status_code=400, detail="Failed to decode image.")

    # 2. Digital Image Preprocessing (applied at full resolution)
    processed = original.copy()

    if denoise:
        processed = denoise_image(processed, kernel=(3, 3))

    if clahe:
        processed = apply_clahe(processed)

    # 3. YOLOv8 Object Detection
    raw_results = detector.detect(processed, conf_threshold=conf)
    detections = detector.extract_detections(raw_results, classifier.class_names)

    # 4. Product Classification & Category Mapping
    classified = classifier.classify_detections(detections)

    # 5. Product Isolation (Cropping ROIs)
    raw_crops = detector.crop_detections(original, detections, padding=8)
    crops_data = []

    for i, (det, crop_img) in enumerate(raw_crops):
        c_res = classified[i] if i < len(classified) else None
        crops_data.append({
            "product_name": c_res.product_name if c_res else det.class_name,
            "category": c_res.category if c_res else "Uncategorized",
            "confidence": round(det.confidence, 4),
            "bbox": det.bbox,
            "image_data": _image_to_base64(crop_img, quality=80),
        })

    # 6. Annotation Visualization
    annotated = detector.draw_bounding_boxes(original, detections)
    annotated_b64 = _image_to_base64(annotated, quality=85)

    # Save outputs to disk
    out_img_dir = OUTPUT_DIR / "detected_images"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img_dir / f"{source_stem}_detected.jpg"), annotated)

    # 7. Statistical Analysis
    stats = compute_stats(classified)
    stats_text = format_stats_report(stats, image_name=source_stem)

    out_rep_dir = OUTPUT_DIR / "reports"
    out_rep_dir.mkdir(parents=True, exist_ok=True)
    (out_rep_dir / f"{source_stem}_stats_report.txt").write_text(stats_text, encoding="utf-8")

    # 8. Charts Generation
    charts_data = {}
    if classified:
        try:
            bar_path = visualizer.bar_chart(classified, stem=source_stem)
            charts_data["bar"] = _file_to_base64(bar_path)
        except Exception as e:
            print(f"Bar chart failed: {e}")

        try:
            pie_path = visualizer.pie_chart(classified, stem=source_stem)
            charts_data["pie"] = _file_to_base64(pie_path)
        except Exception as e:
            print(f"Pie chart failed: {e}")

        try:
            conf_path = visualizer.confidence_distribution(classified, stem=source_stem)
            charts_data["conf"] = _file_to_base64(conf_path)
        except Exception as e:
            print(f"Conf chart failed: {e}")

    # Build category summary for POS receipt
    category_summary = []
    for cat_name, cat_obj in stats.by_category.items():
        category_summary.append({
            "category": cat_name,
            "count": cat_obj.count,
            "share": f"{cat_obj.percentage:.1f}",
            "mean_conf": f"{cat_obj.mean_confidence:.1%}",
            "products": cat_obj.products,
        })

    return {
        "status": "success",
        "total_detections": stats.total_detections,
        "num_categories": stats.num_categories,
        "mean_confidence": stats.mean_confidence,
        "max_confidence": stats.max_confidence,
        "min_confidence": stats.min_confidence,
        "annotated_image": annotated_b64,
        "crops": crops_data,
        "charts": charts_data,
        "category_summary": category_summary,
        "report_text": stats_text,
    }


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting SmartMart AI Checkout web server at http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
