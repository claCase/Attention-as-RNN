from tensorflow_probability.python.math import scan_associative
import tensorflow as tf
from tensorflow.keras import layers, models, activations, losses, backend as tfk
from packaging.version import Version

if Version(tf.__version__) >= Version("2.16.0"):
    from tensorflow.keras.src.layers.rnn.dropout_rnn_cell import (
        DropoutRNNCell as DropCell,
    )
else:
    from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin as DropCell
import numpy as np


class AttentionRNNCell(DropCell, layers.Layer):
    def __init__(
        self,
        heads,
        dim,
        activation="silu",
        concat_heads=False,
        dropout=0.1,
        recurrent_dropout=0.1,
        initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.heads = heads
        self.concat_heads = concat_heads
        self.dim = dim
        self.dropout = max(min(dropout, 1), 0)
        self.recurrent_dropout = max(min(recurrent_dropout, 1), 0)
        self.activation = activations.get(activation)
        self.initializer = initializer
        self.state_size = [
            tf.TensorShape([heads, dim]),  # h
            tf.TensorShape([heads]),  # max
            tf.TensorShape([heads]),  # den
            tf.TensorShape([heads, dim]),  # num
        ]
        self.output_size = tf.TensorShape([heads, dim])

    def build(self, input_shape):
        i = input_shape[-1]
        self.kv_kernel = self.add_weight(
            "kv_kernel",
            shape=(i, self.heads, self.dim, 2),
            initializer=self.initializer,
        )
        self.q_kernel = self.add_weight(
            "q_kernel",
            shape=(self.heads, self.dim),
            initializer=self.initializer,
        )

    @tf.function
    def call(self, inputs, states, training=False):
        h, prev_max, prev_den, prev_num = states

        q = self.q_kernel
        kv = tf.einsum("...i,ihok->...hok", inputs, self.kv_kernel)
        kv = self.activation(kv)

        if self.dropout > 0:
            kv_drop = self.get_dropout_mask_for_cell(
                inputs=kv, training=training, count=1
            )
            kv = kv * kv_drop

        k, v = tf.split(kv, 2, -1)
        k, v = k[..., 0], v[..., 0]

        num, den, cmax = self.recurrence(q, k, v, prev_num, prev_den, prev_max)
        h = num / den[..., None]

        if self.recurrent_dropout > 0:
            h_drop = self.get_recurrent_dropout_mask_for_cell(
                inputs=h, training=training, count=1
            )
            h = h * h_drop

        if self.concat_heads:
            shape = tf.shape(h)
            B = shape[0]
            T = shape[1]
            o = tf.reshape(h, (B, T, -1))
        else:
            o = tf.reduce_sum(h, -2)
        return o, [h, cmax, den, num]

    def recurrence(self, query, key, value, prev_num, prev_den, prev_max):
        """Computes softmax recurrence

        Args:
            query (tf.Tensor): Tensor of shape (B,H,O)
            key (tf.Tensor): Tensor of shape (B,H,O)
            value (tf.Tensor): Tensor of shape (B,H,O)
            prev_num (tf.Tensor): Tensor of shape (B,H,O)
            prev_den (tf.Tensor): Tensor of shape (B,H)
            prev_max (tf.Tensor): Tensor of shape (B,H)

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:  (numerator, denominator, max)
        """
        # Query-Key inner product
        s = tf.einsum("hq,bhq->bh", query, key)  # BH
        # Update max for stable soft-max
        curr_max = tf.maximum(prev_max, s)  # BH
        exp_max_diff = tf.math.exp(prev_max - curr_max)  # BH
        # Subtract max to stabilize
        sm = tf.math.exp(s - curr_max)  # BH
        # Denominator recurrence
        ck = sm + prev_den * exp_max_diff  # BH
        # Numerator recurrence
        ak = value * sm[..., None] + prev_num * exp_max_diff[..., None]  # BHO
        return ak, ck, curr_max


@tf.keras.utils.register_keras_serializable("RNNAttention")
class ScanAssociativeRNNAttention(layers.Layer):
    def __init__(
        self,
        heads,
        dim,
        activation="silu",
        concat_heads=False,
        dropout=0.1,
        recurrent_dropout=0.01,
        initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.heads = heads
        self.concat_heads = concat_heads
        self.dim = dim
        self.dropout = layers.Dropout(max(min(dropout, 1), 0))
        self.recurrent_dropout = layers.Dropout(max(min(recurrent_dropout, 1), 0))
        self.activation = activations.get(activation)
        self.initializer = initializer

    def build(self, input_shape):
        i = input_shape[-1]
        self.kv_kernel = self.add_weight(
            name="kv_kernel",
            shape=(i, self.heads, self.dim, 2),
            initializer=self.initializer,
        )
        self.q_kernel = self.add_weight(
            name="q_kernel", 
            shape=(self.heads, self.dim), 
            initializer=self.initializer
        )

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
        return ua_exp + ub_exp

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
        return wa_exp + wb_exp

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
        m = self.m_aUb(ma0, mb0)  # max
        u = self.u_aUb(ua0, ma0, ub0, mb0)  # denominator
        w = self.w_aUb(wa0, ma0, wb0, mb0)  # numerator
        out = tf.concat([m, u, w], -1)
        return out

    @tf.function
    def call(self, inputs, training):
        """Call function for scan associative operation

        Args:
            inputs (tf.Tensor): (B, T, D)
            training (bool): True if training else False
        """
        return self.scan(inputs, training)[0]

    def scan(self, inputs, training):
        shape = tf.shape(inputs)
        B = shape[0]
        T = shape[1]

        q = self.q_kernel
        kv = tf.einsum("bti,ihok->bthok", inputs, self.kv_kernel)
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
        h = self.recurrent_dropout(h, training=training)
        if self.concat_heads:
            h = tf.reshape(h, (B, T, -1))
        else:
            h = tf.reduce_mean(h, -2)
        return h, m, c, a

    def get_config(self):
        config = {
            "dropout": self.dropout,
            "heads": self.attn_heads,
            "dim": self.dim,
            "concat_heads": self.concat_heads,
            "activation": (
                self.activation
                if type(self.activation) is str
                else tf.keras.utils.serialize_keras_object(self.activation)
            ),
            "initializer": (
                self.initializer
                if type(self.initializer) is str
                else tf.keras.utils.serialize_keras_object(self.initializer)
            ),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Attention(layers.Layer):
    def __init__(
        self,
        heads,
        dims,
        activation,
        dropout,
        return_attention=False,
        causal=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.heads = heads
        self.dims = dims
        self.activation = activations.get(activation)
        self.dropout = layers.Dropout(min(max(dropout, 0), 1))
        self.return_attention = return_attention
        self.causal = causal

    def build(self, input_shape):
        i = input_shape[-1]
        self.attn_kernel = self.add_weight(
            name="kvq_kernel", shape=(i, self.heads, self.dims, 3)
        )
        self.out_kernel = self.add_weight(
            name="kvq_kernel", shape=(self.dims, self.heads, self.dims)
        )

    @tf.function
    def call(self, inputs, training):
        kvq = tf.einsum("bni,ihok->bnhok", inputs, self.attn_kernel)
        kvq = self.dropout(kvq, training=training)

        k, v, q = tf.split(kvq, 3, -1)
        k, v, q = k[..., 0], v[..., 0], q[..., 0]
        qk = tf.einsum("bnho,bkho->bhnk", q, k)
        d = tf.math.sqrt(tf.cast(self.attn_kernel.shape[0], inputs.dtype))
        qk_normed = qk / d
        if self.causal:
            mask = tf.ones_like(qk_normed)
            mask = (
                -(1.0 - tf.linalg.LinearOperatorLowerTriangular(mask).to_dense()) * 1e10
            )
            qk_normed = qk_normed + mask
        A_soft = tf.nn.softmax(qk_normed)
        OH = tf.einsum("bhnk,bkho->bnho", A_soft, v)
        O = tf.einsum("bnhi,iho", OH, self.out_kernel)
        if self.return_attention:
            return O, A_soft
        return O
