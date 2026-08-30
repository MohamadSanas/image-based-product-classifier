from ultralytics import YOLO
from pathlib import Path
import cv2

# =====================================================
# Project paths
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "weights" / "best.pt"
IMAGE_PATH = PROJECT_DIR / "images" / "20180824-13-50-07-6.jpg"

# Output folder inside project
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# =====================================================
# Check files
# =====================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

print("Model :", MODEL_PATH)
print("Image :", IMAGE_PATH)

# =====================================================
# Load model
# =====================================================

model = YOLO(str(MODEL_PATH))
print("✅ Model loaded successfully!")

# =====================================================
# Run prediction
# =====================================================

results = model.predict(
    source=str(IMAGE_PATH),
    imgsz=640,
    conf=0.25,
    save=False,          # Don't save to runs/
    save_txt=False,      # We'll save manually
    verbose=True
)

# =====================================================
# Get prediction image with bounding boxes
# =====================================================

predicted_image = results[0].plot()

# Save image inside project/output/
output_image_path = OUTPUT_DIR / f"{IMAGE_PATH.stem}_prediction.jpg"

cv2.imwrite(str(output_image_path), predicted_image)

# =====================================================
# Save detection text file
# =====================================================

output_txt_path = OUTPUT_DIR / f"{IMAGE_PATH.stem}_prediction.txt"

with open(output_txt_path, "w") as f:

    print("\nDetections:")
    print("-" * 70)

    for box in results[0].boxes:

        cls_id = int(box.cls.item())
        conf = float(box.conf.item())

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        line = (
            f"Class ID: {cls_id}, "
            f"Confidence: {conf:.4f}, "
            f"BBox: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
        )

        print(line)
        f.write(line + "\n")

print("-" * 70)

# =====================================================
# Display image
# =====================================================

cv2.imshow("RPC Product Detection", predicted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# =====================================================
# Output paths
# =====================================================

print("\n✅ Prediction completed!")
print("Saved image :", output_image_path)
print("Saved labels:", output_txt_path)