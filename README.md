# 🛒 Smart Supermarket Product Identification System

An image processing-based smart supermarket checkout system developed for **EC9570 – Digital Image Processing** at the **Faculty of Engineering, University of Jaffna**.

The system detects supermarket products from images, classifies them into predefined product categories using **YOLOv8**, counts detected products, and generates statistical summaries with visualizations.

---

## Features

* Image acquisition and preprocessing.
* Product detection using YOLOv8.
* Product segmentation and isolation.
* Product classification into 200 supermarket product classes.
* Category-wise product counting.
* Statistical analysis and report generation.
* Bar chart and Pie chart visualization.
* Bounding box visualization on detected products.

---

## Technologies

* Python 3.13
* Ultralytics YOLOv8
* OpenCV
* NumPy
* Matplotlib
* PyTorch

---

## Project Structure

```text
smart-supermarket-product-identification/

│

├── data/

│   ├── raw/

│   ├── processed/

│   └── test/

│

├── models/

│   └── weights/

│       └── best.pt

│

├── src/

│   ├── preprocessing.py

│   ├── segmentation.py

│   ├── classifier.py

│   ├── statistics.py

│   ├── visualization.py

│   ├── utils.py

│   ├── predict.py

│   └── main.py

│

├── output/

│   ├── detected_images/

│   ├── reports/

│   └── charts/

│

├── docs/

│

├── requirements.txt

├── README.md

└── .gitignore
```

---

## Workflow

```text
Input Image
      │
      ▼
Image Acquisition & Preprocessing
      │
      ▼
Product Detection & Segmentation
      │
      ▼
Product Classification
      │
      ▼
Product Counting
      │
      ▼
Statistical Analysis
      │
      ▼
Visualization & Report Generation
```

---

## Example Output

```text
Detected Products

Chocolate : 4
Milk      : 2
Drink     : 3
Candy     : 1

Total Products : 10

Distribution

Chocolate   40%
Drink       30%
Milk        20%
Candy       10%
```

The system also generates:

* Annotated images with product bounding boxes.
* Detection report (`.txt`).
* Bar chart of product counts.
* Pie chart showing category distribution.

---

## Team Members & Responsibilities

| Team Member     | Responsibilities                                                                                                 |
| --------------- | ---------------------------------------------------------------------------------------------------------------- |
| **SANAS M.M.**  | **1. Image acquisition and preprocessing** (noise removal, resizing). **3. Product classification module.**      |
| **AHAMED A.A.** | **2. Object detection and segmentation** (product isolation). **4. Statistical analysis and report generation.** |

---

## License

**Educational Use Only**

This project was developed for the **EC9570 – Digital Image Processing** course at the Faculty of Engineering, University of Jaffna.
