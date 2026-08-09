"""
training/trainer.py — Two-phase transfer learning trainer.

Phase 1: Freeze the base model → train only the classification head.
Phase 2: Unfreeze the last N layers → fine-tune at a lower learning rate.
"""
from __future__ import annotations

import logging
from pathlib import Path

import tensorflow as tf

from src.pipeline.classifier import CNNClassifier

logger = logging.getLogger(__name__)


class Trainer:
    """
    Manages the two-phase transfer learning training loop.

    Parameters
    ----------
    classifier : CNNClassifier
        Classifier instance that owns the model architecture definition.
    output_dir : str
        Directory to save best weights and TensorBoard logs.
    epochs_phase1 : int
        Epochs for Phase 1 (frozen base).
    epochs_phase2 : int
        Epochs for Phase 2 (fine-tuning).
    lr_phase1 : float
        Learning rate for Phase 1.
    lr_phase2 : float
        Learning rate for Phase 2 (should be much lower).
    unfreeze_from : int
        Index from which layers are unfrozen for Phase 2. Negative = last N layers.
    """

    def __init__(
        self,
        classifier: CNNClassifier,
        output_dir: str = "models",
        epochs_phase1: int = 10,
        epochs_phase2: int = 20,
        lr_phase1: float = 1e-3,
        lr_phase2: float = 1e-4,
        unfreeze_from: int = -30,
    ) -> None:
        self.classifier = classifier
        self.output_dir = Path(output_dir)
        self.epochs_phase1 = epochs_phase1
        self.epochs_phase2 = epochs_phase2
        self.lr_phase1 = lr_phase1
        self.lr_phase2 = lr_phase2
        self.unfreeze_from = unfreeze_from

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
    ) -> dict:
        """
        Run both training phases and return combined history dictionaries.

        Returns
        -------
        dict with keys "phase1" and "phase2", each being a Keras history dict.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        weights_path = self.output_dir / "weights" / "best_model.keras"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        log_dir = self.output_dir / "logs"

        # ---- Phase 1: Train head only ----------------------------------
        logger.info("=== Phase 1: Training classification head ===")
        model = self.classifier.build(trainable_base=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr_phase1),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary(print_fn=logger.debug)

        callbacks_p1 = self._build_callbacks(weights_path, log_dir / "phase1")
        history_p1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs_phase1,
            callbacks=callbacks_p1,
        )

        # ---- Phase 2: Unfreeze last N layers and fine-tune -------------
        logger.info("=== Phase 2: Fine-tuning last %d layers ===", abs(self.unfreeze_from))
        base_model = model.get_layer(self.classifier.base_model_name.lower())
        base_model.trainable = True
        for layer in base_model.layers[: self.unfreeze_from]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr_phase2),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks_p2 = self._build_callbacks(weights_path, log_dir / "phase2")
        history_p2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs_phase2,
            initial_epoch=self.epochs_phase1,
            callbacks=callbacks_p2,
        )

        logger.info("Training complete. Best model saved to %s", weights_path)

        return {
            "phase1": history_p1.history,
            "phase2": history_p2.history,
            "model_path": str(weights_path),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_callbacks(
        weights_path: Path,
        log_dir: Path,
    ) -> list[tf.keras.callbacks.Callback]:
        return [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(weights_path),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1,
            ),
            tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1),
        ]

    @classmethod
    def from_config(cls, cfg: dict, classifier: CNNClassifier) -> "Trainer":
        t = cfg.get("training", {})
        return cls(
            classifier=classifier,
            output_dir="models",
            epochs_phase1=t.get("epochs_phase1", 10),
            epochs_phase2=t.get("epochs_phase2", 20),
            lr_phase1=t.get("learning_rate_phase1", 1e-3),
            lr_phase2=t.get("learning_rate_phase2", 1e-4),
            unfreeze_from=t.get("unfreeze_from_layer", -30),
        )
