import tensorflow as tf

from .tensor import AbstractTensor
from .prime import PrimeTensor


def binarize(tensor: AbstractTensor, prime: int = 37) -> PrimeTensor:
    with tf.name_scope('binarize'):
        BITS = tensor.int_type.size * 8
        assert prime > BITS, prime

        final_shape = [1] * len(tensor.shape) + [BITS]
        bitwidths = tf.range(BITS, dtype=tf.int32)
        bitwidths = tf.reshape(bitwidths, final_shape)

        val = tf.expand_dims(tensor.value, -1)
        val = tf.bitwise.bitwise_and(tf.bitwise.right_shift(val, bitwidths), 1)

        return PrimeTensor.from_native(val, prime)
