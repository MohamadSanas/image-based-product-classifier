# 🛒 Smart Supermarket Product Identification System

An image processing-based smart supermarket checkout system developed for **EC9570 – Digital Image Processing** at the **Faculty of Engineering, University of Jaffna**.

The system detects supermarket products from images, classifies them into predefined product categories using **YOLOv8**, counts detected products, and generates statistical summaries with visualizations.

---

## Features

* Image acquisition and preprocessing.
* Product detection using YOLOv8.
* Product segmentation and isolation.
* Product classification into **200 supermarket product classes**.
* Category-wise product counting.
* Statistical analysis and report generation.
* Bar chart and Pie chart visualization.
* Bounding box visualization on detected products.

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
* **Household & Personal Care:** Tissue packs, toothpaste, soap, shampoo, and other hygiene products.
* **Stationery:** Pens, notebooks, and other stationery items.

> **Note:** The model predicts **200 individual product classes**, where each class represents a specific supermarket product or package variant within these 17 product categories.

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
├── config/
│   ├── config.yaml
│   └── logging.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── test/
├── docs/
├── images/
├── models/
│   └── weights/
│       └── best.pt
├── notebooks/
├── output/
│   ├── detected_images/
│   ├── reports/
│   └── charts/
├── scripts/
│   └── test.py
├── src/
│   ├── Model_Training/
│   │   ├── Resource_check.py
│   │   ├── dataset_check.py
│   │   └── training_RPC.py
│   └── __init__.py
├── tests/
│   ├── fixtures/
│   ├── integration/
│   └── unit/
├── .env.example
├── pyproject.toml
├── README.md
├── requirements.txt
└── requirements-dev.txt
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

This project was developed for the **EC9570 – Digital Image Processing** course at the **Faculty of Engineering, University of Jaffna**.
