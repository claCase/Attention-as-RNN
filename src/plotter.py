import tensorflow as tf 
from scipy.ndimage import gaussian_filter
import os
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

def plot_hist2d(model, inputs, samples=200, axis=0, resolution=500,save_path=None):
    resolution = max(samples, resolution)
    assert len(inputs.shape) == 2 
    inputs = tf.broadcast_to(inputs, (samples, *inputs.shape))
    samples = max(samples, 4)
    Y = model(inputs, training=True).numpy()[:, :, axis]
    if isinstance(inputs, tf.Tensor):
        inputs = inputs.numpy()
    length = Y.shape[1]
    X = np.linspace(0, 1, length)
    x_fine = np.linspace(0, 1, resolution)
    y_fine = np.concatenate([np.interp(x_fine, X, y_row) for y_row in Y])
    x_fine = np.broadcast_to(x_fine, (samples, resolution)).ravel()
    h, ex, ey = np.histogram2d(x_fine.ravel(), y_fine.ravel(), bins=[100, 50])
    h = gaussian_filter(h, sigma=1)

    fig, ax = plt.subplots()
    ax.pcolormesh(ex, ey, h.T, cmap="inferno")  
    for i in range(4):
        if i == 0:
            ax.plot(X, Y[i], color="white", label="RNN-Attention", alpha=0.5, linewidth=1)
        else:
            ax.plot(X, Y[i], color="white", alpha=0.5, linewidth=1)
    ax.plot(X, inputs[0, :, axis], color="lime",label="True", linewidth=2)
    ax.set_ylim(min(inputs.min(), y_fine.min()), max(inputs.max(), y_fine.max()))
    ax.legend()
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax 