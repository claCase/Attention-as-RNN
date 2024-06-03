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
    DropoutRNNCellMixin, tf.keras.__internal__.layers.BaseRandomLayer
):
    def __init__(
        self,
        heads,
        dim,
        activation,
        concat_heads=False,
        dropout=0.1,
        recurrent_dropout=0.1,
        **kwargs,
    ):
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
        self.kv_kernel = self.add_weight(
            "kv_kernel",
            shape=(i, self.heads, self.dim, 2),
            caching_device=default_caching_device,
        )
        self.q_kernel = self.add_weight(
            "q_kernel",
            shape=(self.heads, self.dim),
            caching_device=default_caching_device,
        )

    # @tf.function
    def call(self, inputs, states, training=False):
        h, prev_num, prev_den, prev_max = states
        prev_max = tf.squeeze(prev_max, -1)
        prev_den = tf.squeeze(prev_den, -1)

        q = self.q_kernel
        kv = tf.einsum("...i,ihok->...hok", inputs, self.kv_kernel)
        kv = self.activation(kv)

        if self.dropout > 0:
            kv_drop = self.get_dropout_mask_for_cell(inputs=kv, training=True, count=1)
            kv = kv * kv_drop

        k, v = tf.split(kv, 2, -1)
        k, v = k[..., 0], v[..., 0]
        num, den, cmax = self.recurrence(q, k, v, prev_num, prev_den, prev_max)
        den = den[..., None]
        cmax = cmax[..., None]
        h = num / den
        if self.recurrent_dropout > 0:
            h_drop = self.get_recurrent_dropout_mask_for_cell(
                inputs=h, training=True, count=1
            )
            h = h * h_drop

        if self.concat_heads:
            B = tf.shape(h)[0]
            o = tf.reshape(h, (B, -1))
        else:
            o = tf.reduce_sum(h, -2)
        return o, [h, num, den, cmax]

    def recurrence(self, query, key, value, prev_num, prev_den, prev_max, cache=None):
        """Computes softmax recurrence

        Args:
            query (tf.Tensor): Tensor of shape (B,H,O)
            key (tf.Tensor): Tensor of shape (B,H,O)
            value (tf.Tensor): Tensor of shape (B,H,O)
            prev_num (tf.Tensor): Tensor of shape (B,H,O)
            prev_den (tf.Tensor): Tensor of shape (B,H)
            prev_max (tf.Tensor): Tensor of shape (B,H)
            cache (tf.Tensor, optional): Cache. Defaults to None.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:  (numerator, denominator, max)
        """
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
        ak = (
            prev_num * exp_max_diff[..., None] + value * sm[..., None]
        )  # BHO
        return ak, ck, curr_max


class ScanAssociativeRNNAttention(layers.Layer):
    def __init__(
        self,
        heads,
        dim,
        activation,
        concat_heads=False,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.heads = heads
        self.concat_heads = concat_heads
        self.dim = dim
        self.dropout = layers.Dropout(max(min(dropout, 1), 0))
        self.activation = activations.get(activation)
        
    def build(self, input_shape):
        i = input_shape[-1]
        self.kv_kernel = self.add_weight(
            name="kv_kernel",
            shape=(i, self.heads, self.dim, 2),
        )
        self.q_kernel = self.add_weight(name="q_kernel", shape=(self.heads, self.dim))

    @staticmethod
    def m(s):
        return tf.reduce_max(s, 1)

    @staticmethod
    def u(s, m):
        sm = s - m[:, None]
        sum_exp_sm = tf.reduce_sum(tf.math.exp(sm), 1)
        return sum_exp_sm

    @staticmethod
    def w(s, m, v):
        sm = s - m[:, None, :]
        sum_exp = tf.reduce_sum(tf.math.exp(sm)[..., None] * v, 1)
        return sum_exp

    @staticmethod
    def m_aUb(a, b):
        """Union operation for max

        Args:
            a (tf.Tensor): (B, T, H, D)
            b (tf.Tensor): (B, T, H, D)

        Returns:
            tf.Tensor: max of shape (B, T, H, D)
        """
        return tf.maximum(a, b)

    def u_aUb(self, ua, ma, ub, mb):
        """Union operation for combining denominators

        Args:
            ua (tf.Tensor): (B, T, H, 1)
            ub (tf.Tensor): (B, T, H, 1)

        Returns:
            tf.Tensor: max of shape (B, T, H, 1)
        """
        m_aub = self.m_aUb(ma, mb)
        ua_exp = ua * tf.math.exp(ma - m_aub)
        ub_exp = ub * tf.math.exp(mb - m_aub)
        return ub_exp + ub_exp

    def w_aUb(self, wa, ma, wb, mb):
        """Union operation for combining numerators

        Args:
            ua (tf.Tensor): (B, T, H, D)
            ub (tf.Tensor): (B, T, H, D)

        Returns:
            tf.Tensor: max of shape (B, T, H, D)
        """
        m_aub = self.m_aUb(ma, mb)
        wa_exp = wa * tf.math.exp(ma - m_aub)
        wb_exp = wb * tf.math.exp(mb - m_aub)
        return wb_exp + wb_exp

    def s(self, q, k):
        return tf.einsum("hd,bthd->bth", q, k)

    def associate(self, a, b):
        """Associative operator ⊕ acting on 3-tuples

        Args:
            a (Tuple[m, w, u]): even indices of split sequence 
            b (Tuple[m, w, u]): odd indices of split sequence  

        Returns:
            c (Tuple[m, w, u]):
        """
        # Split input tensors in m, u, w
        ma0, ua0, wa0 = a[..., :1], a[..., 1:2], a[..., 2:]
        mb0, ub0, wb0 = b[..., :1], b[..., 1:2], b[..., 2:]

        m = self.m_aUb(ma0, mb0)  # max (s)
        u = self.u_aUb(ua0, ma0, ub0, mb0)  # denominator
        w = self.w_aUb(wa0, ma0, wb0, mb0)  # numerator (v)
        out = tf.concat([m, u, w], -1)
        return out

    @tf.function
    def call(self, inputs, training):
        """Call function for scan associative operation

        Args:
            inputs (tf.Tensor): (B, T, D)
            training (bool): True if training else False
        """
        shape = tf.shape(inputs)
        B = shape[0]
        T = shape[1]

        q = self.q_kernel
        kv = tf.einsum("...i,ihok->...hok", inputs, self.kv_kernel)
        kv = self.activation(kv)
        kv = self.dropout(kv, training=training)
        k, v = tf.split(kv, 2, -1)
        k, v = k[..., 0], v[..., 0]

        # Set up for associative scan (prefix sum)
        st = self.s(q, k)[..., None]  # (B, T, H, 1)
        u_init = tf.ones(shape=(B, T, self.heads, 1))
        i = tf.concat([st, u_init, v], -1)
        o = scan_associative(self.associate, i, axis=1)
        m, c, a = o[..., :1], o[..., 1:2], o[..., 2:]
        h = a / c

        if self.concat_heads:
            B = tf.shape(h)[0]
            o = tf.reshape(h, (B, -1))
        else:
            o = tf.reduce_sum(h, -2)
        return o
