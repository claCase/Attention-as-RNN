from typing import List, Tuple, Union
import tensorflow as tf
from tensorflow.keras import models, layers, backend as tfk
import numpy as np
from src.layers import AttentionRNNCell, ScanAssociativeRNNAttention


class AttentionRNN(models.Model):
    def __init__(
        self,
        heads: List,
        dims: List,
        activation="silu",
        output_activation="linear",
        return_sequences=True,
        return_state=False,
        dropout=0.1,
        recurrent_dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dims = dims
        self.heads = heads
        self.activation = activation
        stacked = tf.keras.layers.StackedRNNCells(
            [
                AttentionRNNCell(h, d, activation, False, dropout, recurrent_dropout)
                for h, d in zip(heads, dims)
            ]
        )
        self.rnn = layers.RNN(
            stacked,
            return_sequences=return_sequences,
            return_state=return_state,
        )
        self.dense = tf.keras.layers.Dense(dims[-1], output_activation)

    # @tf.function
    def call(self, inputs, training):
        o = self.rnn(inputs, training=training)
        if self.rnn.return_state is True:
            o, s = o
            return self.dense(o), s
        else:
            return self.dense(o)


class ScanRNNAttentionModel(models.Model):
    def __init__(
        self,
        heads: List,
        dims: List,
        activation="silu",
        output_activation="linear",
        concat_heads=False,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()
        layers = [
            ScanAssociativeRNNAttention(
                heads=head,
                dim=dim,
                activation=activation,
                concat_heads=False,
                dropout=dropout,
            )
            for head, dim in zip(heads[:-1], dims[:-1])
        ]
        layers.append(
            ScanAssociativeRNNAttention(
                heads=heads[-1],
                dim=dims[-1],
                activation=output_activation,
                concat_heads=concat_heads,
                dropout=dropout,
            )
        )
        self.scans = tf.keras.models.Sequential(layers)

    def call(self, inputs, training):
        return self.scans(inputs, training)
