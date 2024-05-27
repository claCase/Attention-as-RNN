from tensorflow_probability.python.math import scan_associative
import tensorflow as tf
from tensorflow.python.keras import layers, models, activations, losses, backend as tfk
from tensorflow.python.keras.layers.recurrent import (
    DropoutRNNCellMixin,
    _config_for_enable_caching_device,
    _caching_device,
)
import numpy as np


class AttentionRNNCell(
    DropoutRNNCellMixin,
    tf.keras.__internal__.layers.BaseRandomLayer
):
    def __init__(self, heads, dim, activation, concat_heads=False,dropout=0.1, recurrent_dropout=0.1,**kwargs):
        super().__init__(**kwargs)
        self.heads = heads
        self.concat_heads = concat_heads
        self.dim = dim
        self.dropout = max(min(dropout, 1), 0)
        self.recurrent_dropout = max(min(recurrent_dropout, 1), 0)
        self.activation = activations.get(activation)
        self.state_size = [
            tf.TensorShape([heads, dim]),  # h
            tf.TensorShape([heads, dim]),  # num
            tf.TensorShape([heads, 1]),  # den
            tf.TensorShape([heads, 1]),  # max
        ]
        self.output_size = tf.TensorShape([heads, dim])

        if tf.compat.v1.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop("enable_caching_device", True)
        else:
            self._enable_caching_device = kwargs.pop("enable_caching_device", False)

    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        i = input_shape[-1]
        self.kvq_kernel = self.add_weight(
            "kvq_kernel", (i, self.heads, self.dim, 3), caching_device=default_caching_device
        )

    @tf.function
    def call(self, inputs, states, training=False):
        h, prev_num, prev_den, prev_max = states
        prev_max = tf.squeeze(prev_max, -1)
        prev_den = tf.squeeze(prev_den, -1)
        kvq = tf.einsum("...i,ihok->...hok", inputs, self.kvq_kernel)
        kvq = self.activation(kvq)

        if self.dropout>0:
            kvq_drop = self.get_dropout_mask_for_cell(inputs=kvq,training=True, count=1)
            kvq = kvq * kvq_drop
        if self.recurrent_dropout>0:
            h_drop = self.get_recurrent_dropout_mask_for_cell(inputs=h, training=True,count=1)
            h = h * h_drop

        k, v, q = tf.split(kvq, 3, -1)
        k, v, q = tf.squeeze(k, -1), tf.squeeze(v, -1), tf.squeeze(q, -1)
        num, den, cmax = self.recurrence(q, k, v, prev_num, prev_den, prev_max)
        den = den[..., None]
        cmax = cmax[..., None]
        h = num / den
        if self.concat_heads:
            B = tf.shape(h)[0]
            o = tf.reshape(h, (B, -1))
        else:
            o = tf.reduce_sum(h, -2)
        return o, [h, num, den, cmax]

    def recurrence(self, query, key, value, prev_num, prev_den, prev_max, cache=None):
        # Query-Key inner product
        s = tf.einsum("...q,...q->...", query, key)  # BH
        # Update max for stable soft-max
        curr_max = tf.maximum(prev_max, s)  # BH
        exp_max_diff = tf.math.exp(prev_max - curr_max)  # BH
        # Subtract max to stabilize 
        sm = tf.math.exp(s - curr_max)  # BH
        # Denominator recurrence 
        ck = prev_den * exp_max_diff + sm  # BH
        # Numerator recurrence 
        ak = prev_num * exp_max_diff[..., None] + value * ck[..., None] # BHO
        return ak, ck, curr_max
