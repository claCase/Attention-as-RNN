from typing import List, Tuple, Union
import tensorflow as tf
from tensorflow.keras import models, layers, backend as tfk
import numpy as np
from src.layers import AttentionRNNCell


class AttentionRNN(models.Model):
    def __init__(
        self, heads:List,dims:List, activation, return_sequences=True, return_state=True, **kwargs
    ):
        super().__init__(**kwargs)
        self.dims = dims
        self.heads = heads
        self.activation = activation
        stacked = tf.keras.layers.StackedRNNCells([AttentionRNNCell(h, d, "relu", False) for h,d in zip(heads,dims)])
        self.rnn = layers.RNN(
            stacked,
            return_sequences=return_sequences,
            return_state=return_state,
        )

    def call(self, inputs, training):
        return self.rnn(inputs, training=training)
