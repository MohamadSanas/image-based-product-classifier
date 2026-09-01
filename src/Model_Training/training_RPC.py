# ============================================================
# 🚀 RPC DATASET - YOLOv8n TRAINING
# 4000 RANDOM TRAIN / 2000 TEST
# 200 PRODUCT CLASSES
# ============================================================

import os
import shutil
import torch
from pathlib import Path

# ------------------------------------------------------------
# 1. INSTALL / CHECK ULTRALYTICS
# ------------------------------------------------------------

try:
    from ultralytics import YOLO
    print("✅ Ultralytics already installed")
except ImportError:
    print("📦 Installing Ultralytics...")
    os.system("pip install -q ultralytics")
    from ultralytics import YOLO
    print("✅ Ultralytics installed")


# ------------------------------------------------------------
# 2. PATHS
# ------------------------------------------------------------

DATASET = Path("/kaggle/working/rpc_4000_2000")
DATA_YAML = DATASET / "data.yaml"

RUN_DIR = Path("/kaggle/working/yolo_runs")
RUN_NAME = "rpc_4000_yolov8n"

print("=" * 70)
print("📁 DATASET CHECK")
print("=" * 70)

print("Dataset :", DATASET)
print("YAML    :", DATA_YAML)
print("Exists  :", DATA_YAML.exists())


# ------------------------------------------------------------
# 3. VERIFY DATASET
# ------------------------------------------------------------

train_images = DATASET / "images" / "train"
train_labels = DATASET / "labels" / "train"

test_images = DATASET / "images" / "test"
test_labels = DATASET / "labels" / "test"

print("\nTrain images :", len(list(train_images.glob("*"))))
print("Train labels :", len(list(train_labels.glob("*.txt"))))

print("Test images  :", len(list(test_images.glob("*"))))
print("Test labels  :", len(list(test_labels.glob("*.txt"))))


# ------------------------------------------------------------
# 4. GPU CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("🚀 GPU CHECK")
print("=" * 70)

print("PyTorch :", torch.__version__)
print("CUDA available :", torch.cuda.is_available())
print("GPU count :", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


# ------------------------------------------------------------
# 5. REMOVE OLD YOLO CACHE FILES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("🧹 CLEANING OLD CACHE")
print("=" * 70)

for cache_file in DATASET.rglob("*.cache"):
    try:
        cache_file.unlink()
        print("Removed:", cache_file)
    except Exception as e:
        print("Could not remove:", cache_file, e)

print("✅ Cache cleanup complete")


# ------------------------------------------------------------
# 6. LOAD YOLOv8 NANO
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("🤖 LOADING YOLOv8n")
print("=" * 70)

model = YOLO("yolov8n.pt")

print("✅ YOLOv8n loaded")


# ------------------------------------------------------------
# 7. TRAIN
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("🔥 STARTING TRAINING")
print("=" * 70)

results = model.train(

    # Dataset
    data=str(DATA_YAML),

    # Model
    pretrained=True,

    # Training
    epochs=30,
    imgsz=640,
    batch=16,

    # IMPORTANT:
    # Use both Tesla T4 GPUs
    device="0,1",

    # Prevent dataset caching from consuming storage
    cache=False,

    # Number of dataloader workers
    workers=2,

    # Validation
    val=True,

    # Save checkpoints
    save=True,
    save_period=5,

    # Early stopping
    patience=10,

    # Augmentation
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,

    # Learning rate
    lr0=0.01,
    lrf=0.01,

    # Output
    project=str(RUN_DIR),
    name=RUN_NAME,

    # Reproducibility
    seed=42,
    deterministic=True,

    # Don't create unnecessary plots/files
    plots=True,

    # Verbose output
    verbose=True
)


# ------------------------------------------------------------
# 8. TRAINING COMPLETE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("🎉 TRAINING COMPLETE")
print("=" * 70)

BEST = RUN_DIR / RUN_NAME / "weights" / "best.pt"
LAST = RUN_DIR / RUN_NAME / "weights" / "last.pt"

print("Best model :", BEST)
print("Exists     :", BEST.exists())

print("Last model :", LAST)
print("Exists     :", LAST.exists())


# ------------------------------------------------------------
# 9. STORAGE CHECK
# ------------------------------------------------------------

def get_size_gb(path):
    total = 0

    if path.is_file():
        return path.stat().st_size / (1024 ** 3)

    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size

    return total / (1024 ** 3)


print("\n" + "=" * 70)
print("💾 STORAGE CHECK")
print("=" * 70)

print(f"Dataset size : {get_size_gb(DATASET):.2f} GB")
print(f"Run size     : {get_size_gb(RUN_DIR):.2f} GB")

usage = shutil.disk_usage("/kaggle/working")

print(f"Used space   : {usage.used / (1024**3):.2f} GB")
print(f"Free space   : {usage.free / (1024**3):.2f} GB")