from tensorflow_probability.python.math import scan_associative
import tensorflow as tf 
from tensorflow.python.keras import layers, models, activations, losses, backend as tfk
import numpy as np 


class AttentionRNN(models.Model):
    def __init__(self, heads, dim, **kwargs):
        super().__init__(**kwargs)
        self.heads = heads 
        self.dim = dim 

    def build(self, input_shape):
        pass 

    def call(self, inputs, training, cache=None, recurrent_mode=True):
        pass 

    def recurrence(self, inputs, cache=None):
        pass 

    