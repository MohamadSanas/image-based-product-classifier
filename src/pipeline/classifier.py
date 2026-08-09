"""
pipeline/classifier.py — TensorFlow/Keras CNN classifier.

Wraps a MobileNetV2 (or configurable base) model and implements IClassifier.
Provides both training-time model building and inference-time prediction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.core.interfaces import IClassifier

logger = logging.getLogger(__name__)


class CNNClassifier(IClassifier):
    """
    Transfer-learning classifier using MobileNetV2 as the feature backbone.

    Architecture:
        Input (224×224×3)
            ↓
        MobileNetV2 (ImageNet weights, optionally frozen)
            ↓
        GlobalAveragePooling2D
            ↓
        Dense(dense_units, relu) + Dropout(dropout_rate)
            ↓
        Dense(n_classes, softmax)

    Parameters
    ----------
    class_names : list[str]
        Ordered list of category names matching the model's output layer.
    confidence_threshold : float
        Minimum confidence to return a prediction; below this, the label
        is reported as "unknown".
    top_k : int
        Number of top predictions to return per call.
    base_model_name : str
        Keras base model identifier: "MobileNetV2", "EfficientNetB0", "ResNet50".
    input_shape : tuple[int, int, int]
    dense_units : int
    dropout_rate : float
    """

    SUPPORTED_BASES = {"MobileNetV2", "EfficientNetB0", "ResNet50"}

    def __init__(
        self,
        class_names: list[str],
        confidence_threshold: float = 0.6,
        top_k: int = 3,
        base_model_name: str = "MobileNetV2",
        input_shape: tuple[int, int, int] = (224, 224, 3),
        dense_units: int = 256,
        dropout_rate: float = 0.3,
    ) -> None:
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.base_model_name = base_model_name
        self.input_shape = input_shape
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self._model: Optional[object] = None  # Keras Model (lazy import)

    # ------------------------------------------------------------------
    # IClassifier implementation
    # ------------------------------------------------------------------

    def load(self, weights_path: str) -> None:
        """Load a previously trained model from disk."""
        import tensorflow as tf  # Lazy import — only when needed

        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(f"Model weights not found at: {path}")

        self._model = tf.keras.models.load_model(str(path))
        logger.info("Loaded model from %s", path)

    def predict(self, roi: np.ndarray) -> list[tuple[str, float]]:
        """
        Classify a single ROI crop.

        Parameters
        ----------
        roi : np.ndarray
            Float32 image normalised to [0,1], shape (H, W, 3).

        Returns
        -------
        list of (label, confidence) sorted by descending confidence.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call .load(weights_path) first.")

        import tensorflow as tf  # noqa: F401

        # Resize to model input if necessary
        if roi.shape[:2] != self.input_shape[:2]:
            import cv2
            roi = cv2.resize(roi, (self.input_shape[1], self.input_shape[0]))

        batch = np.expand_dims(roi, axis=0)  # (1, H, W, 3)
        probabilities = self._model.predict(batch, verbose=0)[0]  # (n_classes,)

        top_k_indices = np.argsort(probabilities)[::-1][: self.top_k]
        results = [
            (self.class_names[i], float(probabilities[i]))
            for i in top_k_indices
        ]
        return results

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Model building (used by training/trainer.py)
    # ------------------------------------------------------------------

    def build(self, trainable_base: bool = False):
        """
        Build and return a new Keras model.

        Parameters
        ----------
        trainable_base : bool
            If True, all base layers are trainable (fine-tuning phase).
            If False, base layers are frozen (head-only training phase).

        Returns
        -------
        tf.keras.Model
        """
        import tensorflow as tf

        base = self._get_base_model()
        base.trainable = trainable_base

        inputs = tf.keras.Input(shape=self.input_shape, name="input")
        x = base(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
        x = tf.keras.layers.Dense(self.dense_units, activation="relu", name="fc1")(x)
        x = tf.keras.layers.Dropout(self.dropout_rate, name="dropout")(x)
        outputs = tf.keras.layers.Dense(self.n_classes, activation="softmax", name="output")(x)

        model = tf.keras.Model(inputs, outputs, name=f"{self.base_model_name}_classifier")
        logger.info(
            "Built %s model: %d classes, trainable_base=%s",
            self.base_model_name, self.n_classes, trainable_base,
        )
        return model

    def _get_base_model(self):
        """Instantiate the chosen Keras base model."""
        import tensorflow as tf

        kwargs = dict(include_top=False, weights="imagenet", input_shape=self.input_shape)
        if self.base_model_name == "MobileNetV2":
            return tf.keras.applications.MobileNetV2(**kwargs)
        elif self.base_model_name == "EfficientNetB0":
            return tf.keras.applications.EfficientNetB0(**kwargs)
        elif self.base_model_name == "ResNet50":
            return tf.keras.applications.ResNet50(**kwargs)
        else:
            raise ValueError(
                f"Unsupported base model: {self.base_model_name}. "
                f"Choose from {self.SUPPORTED_BASES}"
            )

    @classmethod
    def from_config(cls, cfg: dict) -> "CNNClassifier":
        model_cfg = cfg.get("model", {})
        inf_cfg = cfg.get("inference", {})
        categories = cfg.get("categories", [])
        return cls(
            class_names=categories,
            confidence_threshold=inf_cfg.get("confidence_threshold", 0.6),
            top_k=inf_cfg.get("top_k", 3),
            base_model_name=model_cfg.get("base_model", "MobileNetV2"),
            input_shape=tuple(model_cfg.get("input_shape", [224, 224, 3])),
            dense_units=model_cfg.get("dense_units", 256),
            dropout_rate=model_cfg.get("dropout_rate", 0.3),
        )
