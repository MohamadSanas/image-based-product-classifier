# ============================================================
# Module  : Upload & Detect
# Owner   : AHAMED A.A.
# Course  : EC9570 – Digital Image Processing
# Desc    : Opens a file picker so you can choose ANY image from
#           your PC, runs it through the trained YOLO model, then
#           displays the bounding-box image on screen WITH the
#           per-class detection counts overlaid on top of it, and
#           saves both the image and a text summary.
# ============================================================

from ultralytics import YOLO
from pathlib import Path
from collections import Counter
import tkinter as tk
from tkinter import filedialog
import cv2

# =====================================================
# Project paths
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "models" / "weights" / "best.pt"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CONF_THRESHOLD = 0.25
IMGSZ = 640

# =====================================================
# Check model exists
# =====================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")


# =====================================================
# File picker — choose an image from your PC
# =====================================================

def choose_image() -> Path:
    root = tk.Tk()
    root.withdraw()                     # hide the empty root window
    root.attributes("-topmost", True)   # bring the dialog to the front

    selected = filedialog.askopenfilename(
        title="Select an image to classify",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()

    if not selected:
        raise SystemExit("No file selected. Exiting.")

    return Path(selected)


# =====================================================
# Draw a count-summary panel on top of the bounding-box image
# =====================================================

def draw_summary_overlay(image, class_counts: dict, total: int):
    """
    Draws a semi-transparent panel in the top-left corner of *image*
    listing the total detection count and the count for each class.
    Returns a new image (does not mutate the original array).
    """
    lines = [f"Total: {total}"] + [
        f"{name}: {count}" for name, count in class_counts.items()
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    line_height = 24
    padding = 10

    text_widths = [
        cv2.getTextSize(line, font, font_scale, thickness)[0][0]
        for line in lines
    ]
    panel_width = max(text_widths) + padding * 2
    panel_height = line_height * len(lines) + padding * 2

    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
    blended = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

    for i, line in enumerate(lines):
        y = padding + line_height * (i + 1) - 6
        cv2.putText(
            blended, line, (padding, y),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )

    return blended


# =====================================================
# Scale the image down to fit the screen (display only —
# the full-resolution image is still saved to disk untouched)
# =====================================================

def resize_for_display(image, margin: float = 0.85):
    """
    Returns a resized COPY of *image* that fits within the screen,
    preserving aspect ratio. Never upscales small images.
    """
    h, w = image.shape[:2]

    root = tk.Tk()
    root.withdraw()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()

    max_w = int(screen_w * margin)
    max_h = int(screen_h * margin)

    scale = min(max_w / w, max_h / h, 1.0)  # never upscale
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image


IMAGE_PATH = choose_image()

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
    imgsz=IMGSZ,
    conf=CONF_THRESHOLD,
    save=False,
    save_txt=False,
    verbose=False,
)

result = results[0]
class_names = model.names

# =====================================================
# Count detections per class
# =====================================================

detected_names = [
    class_names.get(int(box.cls.item()), "unknown")
    for box in result.boxes
]
class_counts = Counter(detected_names)

# Sort by count, descending
class_counts_sorted = dict(sorted(class_counts.items(), key=lambda kv: -kv[1]))

print("\nDetected classes and counts:")
print("-" * 40)
for name, count in class_counts_sorted.items():
    print(f"{name:<25} x{count}")
print("-" * 40)
print(f"Total detections : {len(result.boxes)}")
print(f"Unique classes    : {len(class_counts_sorted)}")

# =====================================================
# Build bounding-box image WITH the count panel overlaid
# =====================================================

predicted_image = result.plot()
display_image = draw_summary_overlay(
    predicted_image, class_counts_sorted, len(result.boxes)
)

# =====================================================
# Save the annotated + count-overlaid image
# =====================================================

output_image_path = OUTPUT_DIR / f"{IMAGE_PATH.stem}_prediction.jpg"
cv2.imwrite(str(output_image_path), display_image)

# =====================================================
# Save class-count summary as a text file
# =====================================================

output_txt_path = OUTPUT_DIR / f"{IMAGE_PATH.stem}_class_counts.txt"
with open(output_txt_path, "w") as f:
    f.write(f"Image             : {IMAGE_PATH.name}\n")
    f.write(f"Total detections  : {len(result.boxes)}\n")
    f.write(f"Unique classes    : {len(class_counts_sorted)}\n")
    f.write("-" * 40 + "\n")
    for name, count in class_counts_sorted.items():
        f.write(f"{name:<25} x{count}\n")

print("\n✅ Done!")
print("Annotated image :", output_image_path)
print("Class counts    :", output_txt_path)

# =====================================================
# Display the bounding-box image WITH counts on screen
# (scaled to fit your screen — the saved file above is
#  still full resolution)
# =====================================================

screen_image = resize_for_display(display_image)

cv2.namedWindow("Detection Result", cv2.WINDOW_NORMAL)
cv2.imshow("Detection Result", screen_image)
cv2.resizeWindow(
    "Detection Result", screen_image.shape[1], screen_image.shape[0]
)
cv2.waitKey(0)
cv2.destroyAllWindows()