from scipy.ndimage import gaussian_filter
import os
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

def plot_hist2d(model, inputs, samples=200, axis=0, resolution=500):
    assert len(inputs.shape) == 3 and inputs.shape[0] == 1

    samples = max(samples, 4)
    Y = []
    for i in range(samples):
        Y.append(model(inputs, training=True).numpy()[0, :, axis])
    Y = np.asarray(Y)
    length = Y.shape[1]
    X = np.linspace(0, 1, length)
    #X = np.broadcast_to(X, (samples, length))

    x_fine = np.linspace(X.min(), X.max(), resolution)
    y_fine = np.concatenate([np.interp(x_fine, X, y_row) for y_row in Y])
    x_fine = np.broadcast_to(x_fine, (samples, resolution)).ravel()
    h, ex, ey = np.histogram2d(x_fine, y_fine, bins=[100, 50])
    h = gaussian_filter(h, sigma=1)

    fig, ax = plt.subplots()
    ax.pcolormesh(ex, ey, h.T, cmap="inferno")  
    for i in range(4):
        if i == 0:
            ax.plot(X, Y[i], color="blue", label="RNN-Attention", alpha=0.5, linewidth=1)
        else:
            ax.plot(X, Y[i], color="blue", alpha=0.5, linewidth=1)
    ax.plot(X, inputs[0, :, axis], label="True")
    ax.legend()
    return fig, ax 