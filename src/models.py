from typing import List, Tuple, Union
import tensorflow as tf
from tensorflow.keras import models, layers, backend as tfk
import numpy as np
from src.layers import (
    Attention,
    AttentionRNNCell,
    ScanAssociativeRNNAttention,
)


class AttentionRNN(models.Model):
    def __init__(
        self,
        heads: List,
        dims: List,
        activation="silu",
        output_activation="linear",
        concat_heads=False,
        return_sequences=True,
        return_state=False,
        dropout=0.1,
        recurrent_dropout=0.01,
        causal=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dims = dims
        self.heads = heads
        self.activation = activation
        cell = AttentionRNNCell

        stacked = [
            cell(h, d, activation, concat_heads, dropout, recurrent_dropout)
            for h, d in zip(heads[:-1], dims[:-1])
        ]
        stacked.append(
            cell(
                heads[-1],
                dims[-1],
                activation,
                concat_heads,
                dropout,
                recurrent_dropout,
            )
        )
        stackedCell = tf.keras.layers.StackedRNNCells(stacked)
        self.rnn = layers.RNN(
            stackedCell,
            return_sequences=return_sequences,
            return_state=return_state,
        )

    @tf.function
    def call(self, inputs, training):
        return self.rnn(inputs, training=training)


class ScanRNNAttentionModel(models.Model):
    def __init__(
        self,
        heads: List,
        dims: List,
        activation="silu",
        output_activation="linear",
        concat_heads=False,
        dropout=0.1,
        recurrent_dropout=0.01,
        **kwargs,
    ):
        super().__init__()
        assert len(heads) == len(
            dims
        ), f"len of heads and dims must be equal, but are heads:{len(heads)} dims: {len(dims)}"

        layers = [
            ScanAssociativeRNNAttention(
                heads=head,
                dim=dim,
                activation=activation,
                concat_heads=False,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
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
                recurrent_dropout=recurrent_dropout,
            )
        )
        self.scans = tf.keras.models.Sequential(layers)

    def call(self, inputs, training):
        return self.scans(inputs, training)


class AttentionModel(models.Model):
    def __init__(
        self, heads, dims, activation, output_activation, dropout, causal=True, **kwargs
    ):
        super().__init__(**kwargs)
        self.heads = heads
        self.dims = dims
        self.activation = activation
        attn_layers = [
            Attention(head, dim, activation, dropout, False, causal)
            for head, dim in zip(heads[:-1], dims[:-1])
        ]
        attn_layers.append(
            Attention(heads[-1], dims[-1], output_activation, dropout, False, causal)
        )
        self.attn = models.Sequential(attn_layers)

    def call(self, inputs, training):
        return self.attn(inputs, training)
