"""
training/evaluator.py — Model evaluation on a held-out test set.

Generates accuracy, per-class precision/recall/F1, and confusion matrix.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluates a trained Keras model on a test tf.data.Dataset.

    Parameters
    ----------
    class_names : list[str]
        Ordered list of category names.
    """

    def __init__(self, class_names: list[str]) -> None:
        self.class_names = class_names

    def evaluate(
        self,
        model: tf.keras.Model,
        test_ds: tf.data.Dataset,
        output_dir: str = "output/reports",
    ) -> dict:
        """
        Run evaluation and return a metrics dictionary.

        Saves classification report and confusion matrix to output_dir.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        y_true, y_pred = [], []
        for images, labels in test_ds:
            preds = model.predict(images, verbose=0)
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            y_pred.extend(np.argmax(preds, axis=1))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        accuracy = float(np.mean(y_true == y_pred))
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True,
        )
        cm = confusion_matrix(y_true, y_pred)

        logger.info("Test Accuracy: %.4f", accuracy)
        logger.info("\n%s", classification_report(y_true, y_pred, target_names=self.class_names))

        # Persist classification report as text
        report_path = output_path / "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(classification_report(y_true, y_pred, target_names=self.class_names))

        # Persist confusion matrix as CSV
        cm_path = output_path / "confusion_matrix.csv"
        np.savetxt(str(cm_path), cm, delimiter=",", fmt="%d", header=",".join(self.class_names))

        return {
            "accuracy": accuracy,
            "report": report,
            "confusion_matrix": cm.tolist(),
            "report_path": str(report_path),
            "cm_path": str(cm_path),
        }
