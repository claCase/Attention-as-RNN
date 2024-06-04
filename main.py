import tensorflow as tf
import numpy as np
from src import models, layers, utils, plotter
from importlib import reload
import matplotlib.pyplot as plt
import os
from datetime import datetime


if __name__ == "__main__":
    LOG = os.path.join(os.getcwd(), "train_logs")
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(LOG, now)
    length = 100
    x = utils.generate_sin(length=length)
    x = tf.transpose(x, (2, 0, 1))

    x_test = utils.generate_sin()
    x_test = tf.transpose(x_test, (2, 0, 1))
    model = models.ScanRNNAttentionModel(
        heads=[10, 25, 3], dims=[5, 20, 2], activation="silu", concat_heads=False
    )
    _ = model(x)
    model.compile("adam", "mse")
    tb = tf.keras.callbacks.TensorBoard(log_dir, update_freq=1, profile_batch="10, 15")
    check_pt = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(log_dir, "{epoch:02d}.keras"),
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq=200,
        initial_value_threshold=None,
    )
    callbacks = [tb, check_pt]
    history = model.fit(
        x,
        x,
        epochs=400,
        batch_size=50,
        validation_data=(x_test, x_test),
        callbacks=callbacks,
    )
    plotter.plot_hist2d(
        model,
        x[:1],
        save_path=os.path.join(os.getcwd(), "figures", now + "_output_stochastic.png"),
    )
