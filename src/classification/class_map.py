# ============================================================
# Module  : RPC Class Map
# Owner   : SANAS M.M.
# Course  : EC9570 – Digital Image Processing
# Desc    : Provides the ordered list of 204 RPC product class
#           names (index = YOLO class_id) and a lookup mapping
#           each product name to its major product category.
#
# Source  : Retail Product Checkout (RPC) Dataset
#           Wei et al., CVPR 2019
#           https://rpc-dataset.github.io/
# ============================================================

from __future__ import annotations

# ──────────────────────────────────────────────────────────────
# 1. Ordered class name list  (index == YOLO class_id)
# ──────────────────────────────────────────────────────────────
# 200 product classes across 17 major categories.
# NOTE: The RPC dataset contains 204 fine-grained product SKUs across
# 17 major categories (the README table sums to 204).
# The ordering below matches the YOLO training data.yaml used
# in src/Model_Training/training_RPC.py (RPC 4000/2000 split).

RPC_CLASS_NAMES: list[str] = [
    # ── Puffed Food (12) ────────────────────────────────────
    "puffed_food_1",
    "puffed_food_2",
    "puffed_food_3",
    "puffed_food_4",
    "puffed_food_5",
    "puffed_food_6",
    "puffed_food_7",
    "puffed_food_8",
    "puffed_food_9",
    "puffed_food_10",
    "puffed_food_11",
    "puffed_food_12",
    # ── Dried Fruit (9) ─────────────────────────────────────
    "dried_fruit_1",
    "dried_fruit_2",
    "dried_fruit_3",
    "dried_fruit_4",
    "dried_fruit_5",
    "dried_fruit_6",
    "dried_fruit_7",
    "dried_fruit_8",
    "dried_fruit_9",
    # ── Dried Food (9) ──────────────────────────────────────
    "dried_food_1",
    "dried_food_2",
    "dried_food_3",
    "dried_food_4",
    "dried_food_5",
    "dried_food_6",
    "dried_food_7",
    "dried_food_8",
    "dried_food_9",
    # ── Instant Drinks (11) ─────────────────────────────────
    "instant_drink_1",
    "instant_drink_2",
    "instant_drink_3",
    "instant_drink_4",
    "instant_drink_5",
    "instant_drink_6",
    "instant_drink_7",
    "instant_drink_8",
    "instant_drink_9",
    "instant_drink_10",
    "instant_drink_11",
    # ── Instant Noodles (12) ────────────────────────────────
    "instant_noodle_1",
    "instant_noodle_2",
    "instant_noodle_3",
    "instant_noodle_4",
    "instant_noodle_5",
    "instant_noodle_6",
    "instant_noodle_7",
    "instant_noodle_8",
    "instant_noodle_9",
    "instant_noodle_10",
    "instant_noodle_11",
    "instant_noodle_12",
    # ── Desserts (17) ───────────────────────────────────────
    "dessert_1",
    "dessert_2",
    "dessert_3",
    "dessert_4",
    "dessert_5",
    "dessert_6",
    "dessert_7",
    "dessert_8",
    "dessert_9",
    "dessert_10",
    "dessert_11",
    "dessert_12",
    "dessert_13",
    "dessert_14",
    "dessert_15",
    "dessert_16",
    "dessert_17",
    # ── Soft Drinks & Beverages (13) ────────────────────────
    "soft_drink_1",
    "soft_drink_2",
    "soft_drink_3",
    "soft_drink_4",
    "soft_drink_5",
    "soft_drink_6",
    "soft_drink_7",
    "soft_drink_8",
    "soft_drink_9",
    "soft_drink_10",
    "soft_drink_11",
    "soft_drink_12",
    "soft_drink_13",
    # ── Alcoholic Beverages (17) ────────────────────────────
    "alcohol_1",
    "alcohol_2",
    "alcohol_3",
    "alcohol_4",
    "alcohol_5",
    "alcohol_6",
    "alcohol_7",
    "alcohol_8",
    "alcohol_9",
    "alcohol_10",
    "alcohol_11",
    "alcohol_12",
    "alcohol_13",
    "alcohol_14",
    "alcohol_15",
    "alcohol_16",
    "alcohol_17",
    # ── Milk & Dairy Products (12) ──────────────────────────
    "milk_1",
    "milk_2",
    "milk_3",
    "milk_4",
    "milk_5",
    "milk_6",
    "milk_7",
    "milk_8",
    "milk_9",
    "milk_10",
    "milk_11",
    "milk_12",
    # ── Canned Food (14) ────────────────────────────────────
    "canned_food_1",
    "canned_food_2",
    "canned_food_3",
    "canned_food_4",
    "canned_food_5",
    "canned_food_6",
    "canned_food_7",
    "canned_food_8",
    "canned_food_9",
    "canned_food_10",
    "canned_food_11",
    "canned_food_12",
    "canned_food_13",
    "canned_food_14",
    # ── Chocolate (12) ──────────────────────────────────────
    "chocolate_1",
    "chocolate_2",
    "chocolate_3",
    "chocolate_4",
    "chocolate_5",
    "chocolate_6",
    "chocolate_7",
    "chocolate_8",
    "chocolate_9",
    "chocolate_10",
    "chocolate_11",
    "chocolate_12",
    # ── Chewing Gum (8) ─────────────────────────────────────
    "chewing_gum_1",
    "chewing_gum_2",
    "chewing_gum_3",
    "chewing_gum_4",
    "chewing_gum_5",
    "chewing_gum_6",
    "chewing_gum_7",
    "chewing_gum_8",
    # ── Candy (10) ──────────────────────────────────────────
    "candy_1",
    "candy_2",
    "candy_3",
    "candy_4",
    "candy_5",
    "candy_6",
    "candy_7",
    "candy_8",
    "candy_9",
    "candy_10",
    # ── Seasonings / Spices (12) ────────────────────────────
    "seasoning_1",
    "seasoning_2",
    "seasoning_3",
    "seasoning_4",
    "seasoning_5",
    "seasoning_6",
    "seasoning_7",
    "seasoning_8",
    "seasoning_9",
    "seasoning_10",
    "seasoning_11",
    "seasoning_12",
    # ── Personal Hygiene Products (10) ──────────────────────
    "personal_hygiene_1",
    "personal_hygiene_2",
    "personal_hygiene_3",
    "personal_hygiene_4",
    "personal_hygiene_5",
    "personal_hygiene_6",
    "personal_hygiene_7",
    "personal_hygiene_8",
    "personal_hygiene_9",
    "personal_hygiene_10",
    # ── Tissue Products (19) ────────────────────────────────
    "tissue_1",
    "tissue_2",
    "tissue_3",
    "tissue_4",
    "tissue_5",
    "tissue_6",
    "tissue_7",
    "tissue_8",
    "tissue_9",
    "tissue_10",
    "tissue_11",
    "tissue_12",
    "tissue_13",
    "tissue_14",
    "tissue_15",
    "tissue_16",
    "tissue_17",
    "tissue_18",
    "tissue_19",
    # ── Stationery (7) ──────────────────────────────────────
    "stationery_1",
    "stationery_2",
    "stationery_3",
    "stationery_4",
    "stationery_5",
    "stationery_6",
    "stationery_7",
]

assert len(RPC_CLASS_NAMES) == 204, (
    f"RPC_CLASS_NAMES must contain exactly 204 entries; got {len(RPC_CLASS_NAMES)}"
)

# ──────────────────────────────────────────────────────────────
# 2. The 17 major product categories
# ──────────────────────────────────────────────────────────────

RPC_CATEGORIES: list[str] = [
    "Puffed Food",
    "Dried Fruit",
    "Dried Food",
    "Instant Drinks",
    "Instant Noodles",
    "Desserts",
    "Soft Drinks & Beverages",
    "Alcoholic Beverages",
    "Milk & Dairy Products",
    "Canned Food",
    "Chocolate",
    "Chewing Gum",
    "Candy",
    "Seasonings / Spices",
    "Personal Hygiene Products",
    "Tissue Products",
    "Stationery",
]

# ──────────────────────────────────────────────────────────────
# 3. Product name → major category lookup
# ──────────────────────────────────────────────────────────────
# Built programmatically from the category boundaries above so
# that it stays in sync with RPC_CLASS_NAMES automatically.

_CATEGORY_SIZES: list[tuple[str, int]] = [
    ("Puffed Food",              12),
    ("Dried Fruit",               9),
    ("Dried Food",                9),
    ("Instant Drinks",           11),
    ("Instant Noodles",          12),
    ("Desserts",                 17),
    ("Soft Drinks & Beverages",  13),
    ("Alcoholic Beverages",      17),
    ("Milk & Dairy Products",    12),
    ("Canned Food",              14),
    ("Chocolate",                12),
    ("Chewing Gum",               8),
    ("Candy",                    10),
    ("Seasonings / Spices",      12),
    ("Personal Hygiene Products",10),
    ("Tissue Products",          19),
    ("Stationery",                7),
]

assert sum(n for _, n in _CATEGORY_SIZES) == 204, (
    "Category sizes must sum to 204."
)


def _build_category_map() -> dict[str, str]:
    """Build the product-name → category dict from ``_CATEGORY_SIZES``."""
    mapping: dict[str, str] = {}
    idx = 0
    for category, count in _CATEGORY_SIZES:
        for _ in range(count):
            mapping[RPC_CLASS_NAMES[idx]] = category
            idx += 1
    return mapping


RPC_CATEGORY_MAP: dict[str, str] = _build_category_map()
"""Maps each of the 204 product class names to its major category string."""


# ──────────────────────────────────────────────────────────────
# 4. Convenience helper
# ──────────────────────────────────────────────────────────────

def get_category(class_name: str, default: str = "Unknown") -> str:
    """
    Return the major category for a given RPC *class_name*.

    Parameters
    ----------
    class_name : str
        A product class name string (element of ``RPC_CLASS_NAMES``).
    default : str
        Value returned when *class_name* is not found in the map.
        Defaults to ``"Unknown"``.

    Returns
    -------
    str
        Major category string (one of ``RPC_CATEGORIES``), or *default*.
    """
    return RPC_CATEGORY_MAP.get(class_name, default)
