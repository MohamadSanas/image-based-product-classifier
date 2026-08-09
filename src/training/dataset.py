"""
training/dataset.py — tf.data pipeline with augmentation.

Loads a directory-structured dataset (one sub-folder per class) and returns
(train_ds, val_ds) tf.data.Dataset objects, ready for model.fit().
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Builds tf.data.Dataset pipelines from an image folder structured as:

        data/raw/
            apple/
                img1.jpg
                img2.jpg
            orange/
                ...

    Parameters
    ----------
    data_dir : str
        Root directory of the dataset.
    image_size : tuple[int, int]
        (height, width) to resize images to.
    batch_size : int
    validation_split : float
        Fraction of data used for validation.
    seed : int
        Random seed for reproducibility.
    augment : bool
        Apply data augmentation on the training split.
    """

    def __init__(
        self,
        data_dir: str,
        image_size: tuple[int, int] = (224, 224),
        batch_size: int = 32,
        validation_split: float = 0.2,
        seed: int = 42,
        augment: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.seed = seed
        self.augment = augment

        self._class_names: list[str] = []

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    def build(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Build and return (train_ds, val_ds).

        Returns
        -------
        tuple of (train_dataset, validation_dataset)
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        train_ds = tf.keras.utils.image_dataset_from_directory(
            str(self.data_dir),
            validation_split=self.validation_split,
            subset="training",
            seed=self.seed,
            image_size=self.image_size,
            batch_size=self.batch_size,
            label_mode="categorical",
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            str(self.data_dir),
            validation_split=self.validation_split,
            subset="validation",
            seed=self.seed,
            image_size=self.image_size,
            batch_size=self.batch_size,
            label_mode="categorical",
        )

        self._class_names = train_ds.class_names
        logger.info(
            "Dataset loaded: %d classes, batch_size=%d",
            len(self._class_names), self.batch_size,
        )

        # Normalise pixel values to [0, 1]
        normalise = tf.keras.layers.Rescaling(1.0 / 255)
        train_ds = train_ds.map(lambda x, y: (normalise(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(lambda x, y: (normalise(x), y), num_parallel_calls=tf.data.AUTOTUNE)

        if self.augment:
            augmentor = self._build_augmentor()
            train_ds = train_ds.map(
                lambda x, y: (augmentor(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        train_ds = train_ds.cache().shuffle(1000, seed=self.seed).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds

    @staticmethod
    def _build_augmentor() -> tf.keras.Sequential:
        """Return a Sequential layer of augmentation transforms."""
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomBrightness(factor=0.2),
            tf.keras.layers.RandomContrast(factor=0.2),
        ], name="augmentation")

    @classmethod
    def from_config(cls, cfg: dict) -> "DatasetBuilder":
        pre = cfg.get("preprocessing", {})
        train_cfg = cfg.get("training", {})
        return cls(
            data_dir=cfg["paths"]["data_raw"],
            image_size=tuple(pre.get("target_size", [224, 224])),
            batch_size=train_cfg.get("batch_size", 32),
            validation_split=train_cfg.get("validation_split", 0.2),
            seed=train_cfg.get("seed", 42),
            augment=True,
        )
