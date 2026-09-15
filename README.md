# 🛒 Smart Supermarket Product Identification System

An image processing-based smart supermarket checkout system developed for **EC9570 – Digital Image Processing** at the **Faculty of Engineering, University of Jaffna**.

The system detects supermarket products from images, classifies them into predefined product categories using **YOLOv8**, counts detected products, and generates statistical summaries with rich visualizations.

---

## Features

* **Interactive File Picker & Overlay:** Select any image from your system, run real-time YOLOv8 detection, display bounding boxes with a semi-transparent class-count panel, and fit to screen size automatically.
* **Image Acquisition & Preprocessing:** Noise reduction, contrast adjustment, and resolution normalization.
* **Product Detection & Segmentation:** High-precision bounding box localization using fine-tuned **YOLOv8**.
* **Product Classification:** Classifies items into **200 supermarket product classes** across 17 main categories.
* **Category-wise & Item-wise Counting:** Automatic counting and summary aggregation.
* **Batch Processing Engine:** Process multiple test images in a single run with re-used GPU/CPU model memory.
* **Statistical Analysis & Visualization:** Automatic generation of detection reports (`.txt`), summary CSVs (`.csv`), bar charts, pie charts, and confidence distribution histograms.

---

## Product Categories Detected

The trained **YOLOv8** model detects and classifies **200 supermarket products** belonging to **17 major product categories** from the **Retail Product Checkout (RPC)** dataset.

| Product Category          | Number of Product Classes |
| ------------------------- | ------------------------: |
| Puffed Food               |                        12 |
| Dried Fruit               |                         9 |
| Dried Food                |                         9 |
| Instant Drinks            |                        11 |
| Instant Noodles           |                        12 |
| Desserts                  |                        17 |
| Soft Drinks & Beverages   |                        13 |
| Alcoholic Beverages       |                        17 |
| Milk & Dairy Products     |                        12 |
| Canned Food               |                        14 |
| Chocolate                 |                        12 |
| Chewing Gum               |                         8 |
| Candy                     |                        10 |
| Seasonings / Spices       |                        12 |
| Personal Hygiene Products |                        10 |
| Tissue Products           |                        19 |
| Stationery                |                         7 |

**Examples of products detected include:**
* **Beverages:** Coca-Cola, Pepsi, bottled drinks, juice, milk cartons, coffee drinks.
* **Snacks:** Chips, puffed food, chocolates, candies, chewing gum.
* **Instant Foods:** Instant noodles, soup mixes, instant drink powders.
* **Groceries:** Dried fruits, canned food, seasonings, and spices.
* **Household & Personal Care:** Tissue packs, toothpaste, soap, shampoo, and hygiene products.
* **Stationery:** Pens, notebooks, and stationery items.

---

## Technologies

* **Python 3.13 / 3.10+**
* **Ultralytics YOLOv8**
* **OpenCV (`cv2`)**
* **Tkinter** (Interactive GUI dialogs)
* **NumPy & Pandas**
* **Matplotlib** (Chart generation)
* **PyTorch**

---

## Project Structure

```text
image-based-product-classifier/
├── config/
│   ├── config.yaml                     # System pipeline & inference configuration
│   └── logging.yaml                    # System logging configuration
├── data/
│   ├── raw/                            # Raw dataset storage
│   ├── processed/                      # Preprocessed data storage
│   └── test/                           # Test dataset partition
├── docs/                               # Project documentation
├── images/                             # Sample & uploaded test images
├── models/
│   └── weights/
│       └── best.pt                     # Fine-tuned YOLOv8 model weights
├── notebook/
│   └── grocery-product.ipynb           # Model training & evaluation notebook
├── output/
│   ├── detected_images/                # Saved bounding-box annotated images
│   ├── reports/                        # Detection summaries & CSV reports
│   └── charts/                         # Bar, pie, and confidence charts
├── scripts/
│   ├── test.py                         # Single-image quick test runner
│   └── batch_run.py                    # Batch runner for all images in images/
├── src/
│   ├── classification/                 # Classification & counting logic
│   ├── detection/                      # YOLOv8 ProductDetector class & bbox logic
│   ├── Model_Training/                 # Model training & dataset check scripts
│   ├── preprocessing/                  # Image acquisition & filtering pipeline
│   ├── reporting/                      # Visualizer & statistical report generators
│   ├── config_loader.py                # YAML configuration loader
│   └── __init__.py
├── tests/                              # Unit & integration tests
├── final_image_classification_module.py# Main GUI File Picker & Detection Module
├── run.py                              # Main CLI pipeline entry point
├── .env.example
├── pyproject.toml
├── README.md
├── requirements.txt
└── requirements-dev.txt
```

---

## Usage & Execution

### 1. Interactive Classification GUI (Choose Any Image)
Run the standalone final classification module to select an image from your PC using a graphical file chooser dialog. It displays the annotated results on screen with an overlay count panel and auto-scales to your monitor resolution:

```bash
python final_image_classification_module.py
```

### 2. Full End-to-End Pipeline CLI
Run the complete pipeline (preprocessing, detection, classification, statistical report, and chart creation) for a specific image:

```bash
# Process a single image
python run.py --image images/20180824-13-43-33-401.jpg

# Adjust confidence threshold
python run.py --image images/20180824-13-43-33-401.jpg --conf 0.35
```

### 3. Batch Image Processing
Process all images located inside the `images/` directory at once, saving individual annotated images, reports, charts, and a master `batch_detection_summary.csv`:

```bash
python scripts/batch_run.py
```

### 4. Direct YOLO Test Script
Quick single-image detection test:

```bash
python scripts/test.py
```

---

## Example Output

```text
Detected Products
----------------------------------------
Desserts                           : 4
Soft Drinks & Beverages            : 2
Alcoholic Beverages                : 1
Instant Noodles                    : 1
----------------------------------------
Total Products : 8

Distribution
----------------------------------------
Desserts                             50.0%
Soft Drinks & Beverages              25.0%
Alcoholic Beverages                  12.5%
Instant Noodles                      12.5%
```

The system generates:
* **Annotated Images:** Bounding boxes and labels on detected products (`output/detected_images/`).
* **Text & Statistical Reports:** Summary reports and detection details (`output/reports/`).
* **Graphical Charts:** Bar charts, pie charts, and confidence distribution graphs (`output/charts/`).

---

## Team Members & Responsibilities

| Team Member     | Responsibilities                                                                                                 |
| --------------- | ---------------------------------------------------------------------------------------------------------------- |
| **SANAS M.M.**  | **1. Image acquisition and preprocessing** (noise removal, resizing). **3. Product classification module.**      |
| **AHAMED A.A.** | **2. Object detection and segmentation** (product isolation). **4. Statistical analysis and report generation.** |

---

## License

**Educational Use Only**

This project was developed for the **EC9570 – Digital Image Processing** course at the **Faculty of Engineering, University of Jaffna**.
