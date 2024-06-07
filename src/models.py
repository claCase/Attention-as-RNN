from typing import List, Tuple, Union
import tensorflow as tf
from tensorflow.keras import models, layers, backend as tfk
import numpy as np
from src.layers import (
    Attention,
    AttentionRNNCell,
    ScanAssociativeRNNAttention,
    LinearSelfAttentionRNN,
    LinearSelfAttention,
)


@tf.keras.utils.register_keras_serializable("RNNAttention")
class AttentionRNN(models.Model):
    def __init__(
        self,
        heads: List[int],
        dims: List[int],
        activation="silu",
        output_activation="linear",
        concat_heads=False,
        return_sequences=True,
        return_state=False,
        dropout=0.1,
        recurrent_dropout=0.01,
        initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dims = dims
        self.heads = heads
        self.activation = activation

        cell = AttentionRNNCell
        stacked = [
            cell(
                h, d, activation, concat_heads, dropout, recurrent_dropout, initializer
            )
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
                initializer,
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


@tf.keras.utils.register_keras_serializable("RNNAttention")
class ScanRNNAttentionModel(models.Model):
    def __init__(
        self,
        heads: List[int],
        dims: List[int],
        activation="silu",
        output_activation="linear",
        concat_heads=False,
        dropout=0.1,
        recurrent_dropout=0.01,
        initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
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
                initializer=initializer,
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
                initializer=initializer,
            )
        )
        self.scans = tf.keras.models.Sequential(layers)

    def call(self, inputs, training):
        h = self.scans(inputs, training=training)
        return h


            )
        )
        self.scans = tf.keras.models.Sequential(layers)

    def call(self, inputs, training):
        return self.scans(inputs, training)


@tf.keras.utils.register_keras_serializable("RNNAttention")
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


@tf.keras.utils.register_keras_serializable("RNNAttention")
class LinearRNNAttentionModel(models.Model):
    def __init__(
        self,
        heads: List[int],
        dims: List[int],
        activation="silu",
        output_activation="linear",
        concat_heads=False,
        dropout=0.1,
        recurrent_dropout=0.01,
        return_state=False,
        return_sequences=True,
        initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert len(heads) == len(
            dims
        ), f"len of heads and dims must be equal, but are heads:{len(heads)} dims: {len(dims)}"

        stacked = [
            LinearSelfAttentionRNN(
                heads=head,
                dims=dim,
                activation=activation,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                initializer=initializer,
            )
            for head, dim in zip(heads[:-1], dims[:-1])
        ]
        stacked.append(
            LinearSelfAttentionRNN(
                heads=heads[-1],
                dims=dims[-1],
                activation=output_activation,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                initializer=initializer,
            )
        )
        stackedCell = tf.keras.layers.StackedRNNCells(stacked)
        self.rnn = layers.RNN(
            stackedCell,
            return_sequences=return_sequences,
            return_state=return_state,
        )

    def call(self, inputs, training):
        return self.rnn(inputs, training=training)


@tf.keras.utils.register_keras_serializable("RNNAttention")
class LinearAttentionModel(models.Model):
    def __init__(
        self,
        heads: List[int],
        dims: List[int],
        activation="silu",
        output_activation="linear",
        dropout=0.1,
        recurrent_dropout=0.01,
        initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert len(heads) == len(
            dims
        ), f"len of heads and dims must be equal, but are heads:{len(heads)} dims: {len(dims)}"

        layers = [
            LinearSelfAttention(
                heads=head,
                dims=dim,
                activation=activation,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                initializer=initializer,
            )
            for head, dim in zip(heads[:-1], dims[:-1])
        ]
        layers.append(
            LinearSelfAttention(
                heads=heads[-1],
                dims=dims[-1],
                activation=output_activation,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                initializer=initializer,
            )
        )
        self.attn = tf.keras.models.Sequential(layers)

    def call(self, inputs, training):
        h = self.attn(inputs, training=training)
        return h
