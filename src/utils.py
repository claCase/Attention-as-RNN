import numpy as np 

def generate_sin(thetas=(0.1, 1), length=100, samples=50, noise=0.01):
    thetas = np.asarray(thetas)
    noise = np.random.normal(size=(len(thetas), samples)) * noise
    x = np.linspace(np.zeros(len(thetas)), np.ones(len(thetas)) * length, length)[
        :, :, None
    ] * (
        thetas[None, :, None] + noise[None, :, :]
    )  #  length x clusters
    return np.sin(x)