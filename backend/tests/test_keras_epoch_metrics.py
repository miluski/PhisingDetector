import numpy as np
import tensorflow as tf

from utils.keras_epoch_metrics import EpochEndEvaluationCallback


def test_epoch_end_metrics_use_completed_epoch_weights():
    tf.keras.utils.set_random_seed(42)
    x = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    y = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(2,)),
            tf.keras.layers.Dense(4, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    callback = EpochEndEvaluationCallback(x, y)

    history = model.fit(
        x,
        y,
        validation_data=(x, y),
        epochs=2,
        batch_size=2,
        callbacks=[callback],
        verbose=0,
    )

    assert len(callback.loss) == 2
    assert len(callback.accuracy) == 2
    assert len(callback.auc) == 2
    assert len(history.history["val_loss"]) == 2
    assert np.isfinite(callback.loss).all()
