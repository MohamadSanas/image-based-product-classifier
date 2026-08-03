# 🛒 Smart Supermarket Product Identification System

An image processing based smart supermarket checkout system developed for **EC9570 – Digital Image Processing** at the **Faculty of Engineering, University of Jaffna**.

The system detects supermarket products from images, classifies them into predefined categories, counts each category, and generates statistical summaries.

---

## Features

- Image preprocessing
- Product detection
- Product segmentation
- Product classification
- Category-wise counting
- Statistical report generation
- Bar chart & Pie chart visualization
- Bounding box visualization

---

## Technologies

- Python 3.13
- OpenCV
- NumPy
- Matplotlib
- Scikit-Learn
- TensorFlow

---

## Project Structure

```
smart-supermarket-product-identification/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── test/
│
├── models/
│
├── src/
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── classifier.py
│   ├── statistics.py
│   ├── visualization.py
│   ├── utils.py
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

```
Input Image
      │
      ▼
Preprocessing
      │
      ▼
Segmentation
      │
      ▼
Feature Extraction
      │
      ▼
Classification
      │
      ▼
Counting
      │
      ▼
Statistics
      │
      ▼
Visualization
```

---

## Example Output

```
Detected Products

Apple          : 3
Orange         : 2
Milk           : 1
Chocolate      : 4

Total Products : 10

Distribution

Apple       30%
Orange      20%
Milk        10%
Chocolate   40%
```

---

## Team Members

| Name | Responsibilities |
|------|------------------|
| Member 01 | Image preprocessing, segmentation |
| Member 02 | Classification, statistics, visualization |

---

## License

Educational Use Only
