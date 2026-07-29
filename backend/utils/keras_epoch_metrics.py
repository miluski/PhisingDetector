"""Keras callbacks used by MLP experiment notebooks."""

from __future__ import annotations

import tensorflow as tf


class EpochEndEvaluationCallback(tf.keras.callbacks.Callback):
    """Evaluate training data in inference mode after every completed epoch.

    Keras batch training metrics are accumulated while weights change and
    regularization layers such as dropout are active.  Validation metrics are
    evaluated after the epoch in inference mode.  This callback gives training
    metrics the same timing and inference semantics as validation metrics.
    """

    def __init__(self, x_train, y_train, x_test=None, y_test=None):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.loss: list[float] = []
        self.accuracy: list[float] = []
        self.auc: list[float] = []
        self.test_accuracy: list[float] = []

    def on_epoch_end(self, epoch, logs=None):
        """Record completed-epoch metrics and resource samples."""
        metrics = self.model.evaluate(
            self.x_train,
            self.y_train,
            verbose=0,
            return_dict=True,
        )
        self.loss.append(float(metrics["loss"]))
        self.accuracy.append(float(metrics["accuracy"]))
        self.auc.append(float(metrics["auc"]))
        if self.x_test is not None and self.y_test is not None:
            test_metrics = self.model.evaluate(
                self.x_test,
                self.y_test,
                verbose=0,
                return_dict=True,
            )
            self.test_accuracy.append(float(test_metrics["accuracy"]))
