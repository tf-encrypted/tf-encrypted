# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.protocol import ABY3
from tf_encrypted.protocol import Pond
from tf_encrypted.protocol import SecureNN

# set tfe protocol
# tfe.set_protocol(tfe.protocol.Pond())
# tfe.set_protocol(tfe.protocol.SecureNN())
tfe.set_protocol(tfe.protocol.ABY3())


@unittest.skipIf(isinstance(tfe.get_protocol(), ABY3), "ABY3 not support")
class TestBatchToSpaceND(unittest.TestCase):
    def setUp(self):
        tf.keras.utils.set_random_seed(4224)

    def test_4d_no_crops(self):
        backing = [
            [[[1], [3]], [[9], [11]]],
            [[[2], [4]], [[10], [12]]],
            [[[5], [7]], [[13], [15]]],
            [[[6], [8]], [[14], [16]]],
        ]
        t = tf.constant(backing)
        block_shape = [2, 2]
        crops = [[0, 0], [0, 0]]
        self._generic_private_test(t, block_shape, crops)

    def test_4d_single_crop(self):
        backing = [
            [[[0], [1], [3]]],
            [[[0], [9], [11]]],
            [[[0], [2], [4]]],
            [[[0], [10], [12]]],
            [[[0], [5], [7]]],
            [[[0], [13], [15]]],
            [[[0], [6], [8]]],
            [[[0], [14], [16]]],
        ]
        t = tf.constant(backing)
        block_shape = [2, 2]
        crops = [[0, 0], [2, 0]]
        self._generic_private_test(t, block_shape, crops)

    def test_3d_no_crops(self):
        t = tf.random.uniform([16, 20, 10])  # e.g. [batch, time, features]
        block_shape = [4]
        crops = [[0, 0]]
        self._generic_private_test(t, block_shape, crops)

    def test_3d_mirror_crops(self):
        t = tf.random.uniform([16, 20, 10])  # e.g. [batch, time, features]
        block_shape = [4]
        crops = [[2, 2]]
        self._generic_private_test(t, block_shape, crops)

    def test_3d_uneven_crops(self):
        t = tf.random.uniform([16, 20, 10])  # e.g. [batch, time, features]
        block_shape = [4]
        crops = [[2, 0]]
        self._generic_private_test(t, block_shape, crops)

    def test_3d_block_shape(self):
        t = tf.random.uniform([16, 20, 10])  # e.g. [batch, time, features]
        block_shape = [8]
        crops = [[0, 0]]
        self._generic_private_test(t, block_shape, crops)

    def test_public(self):
        t = tf.random.uniform([16, 20, 10])  # e.g. [batch, time, features]
        block_shape = [4]
        crops = [[2, 2]]
        self._generic_public_test(t, block_shape, crops)

    def test_masked(self):
        t = tf.random.uniform([16, 20, 10])  # e.g. [batch, time, features]
        block_shape = [4]
        crops = [[2, 2]]
        self._generic_masked_test(t, block_shape, crops)

    @staticmethod
    def _generic_public_test(t, block_shape, crops):
        actual = tf.batch_to_space(t, block_shape=block_shape, crops=crops)
        b = tfe.define_public_variable(t)
        final = tfe.batch_to_space(b, block_shape=block_shape, crops=crops).to_native()
        np.testing.assert_array_almost_equal(final, actual, decimal=3)

    @staticmethod
    def _generic_private_test(t, block_shape, crops):
        actual = tf.batch_to_space(t, block_shape=block_shape, crops=crops)
        b = tfe.define_private_variable(t)
        out = tfe.batch_to_space(b, block_shape=block_shape, crops=crops)
        final = out.reveal().to_native()
        np.testing.assert_array_almost_equal(final, actual, decimal=3)

    @staticmethod
    def _generic_masked_test(t, block_shape, crops):
        actual = tf.batch_to_space(t, block_shape=block_shape, crops=crops)
        b = tfe.mask(tfe.define_private_variable(t))
        out = tfe.batch_to_space(b, block_shape=block_shape, crops=crops)
        final = out.reveal().to_native()
        np.testing.assert_array_almost_equal(final, actual, decimal=3)


@unittest.skipIf(isinstance(tfe.get_protocol(), ABY3), "ABY3 not support")
class TestSpaceToBatchND(unittest.TestCase):
    def setUp(self):
        tf.keras.utils.set_random_seed(4224)

    def test_4d_no_crops(self):
        backing = [
            [[[1], [3]], [[9], [11]]],
            [[[2], [4]], [[10], [12]]],
            [[[5], [7]], [[13], [15]]],
            [[[6], [8]], [[14], [16]]],
        ]
        t = tf.constant(backing)
        block_shape = [2, 2]
        paddings = [[0, 0], [0, 0]]
        self._generic_private_test(t, block_shape, paddings)

    def test_4d_single_crop(self):
        backing = [
            [[[1], [2], [3], [4]], [[5], [6], [7], [8]]],
            [[[9], [10], [11], [12]], [[13], [14], [15], [16]]],
        ]
        t = tf.constant(backing)
        block_shape = [2, 2]
        paddings = [[0, 0], [2, 0]]
        self._generic_private_test(t, block_shape, paddings)

    def test_3d_no_crops(self):
        t = tf.random.uniform([16, 20, 10])  # e.g. [batch, time, features]
        block_shape = [4]
        paddings = [[0, 0]]
        self._generic_private_test(t, block_shape, paddings)

    def test_3d_mirror_crops(self):
        t = tf.random.uniform([16, 20, 10])  # e.g. [batch, time, features]
        block_shape = [4]
        paddings = [[2, 2]]
        self._generic_private_test(t, block_shape, paddings)

    def test_3d_uneven_crops(self):
        t = tf.random.uniform([16, 20, 10])  # e.g. [batch, time, features]
        block_shape = [2]
        paddings = [[2, 0]]
        self._generic_private_test(t, block_shape, paddings)

    def test_3d_block_shape(self):
        t = tf.random.uniform([16, 20, 10])  # e.g. [batch, time, features]
        block_shape = [5]
        paddings = [[0, 0]]
        self._generic_private_test(t, block_shape, paddings)

    def test_public(self):
        t = tf.random.uniform([16, 20, 10])  # e.g. [batch, time, features]
        block_shape = [4]
        paddings = [[2, 2]]
        self._generic_public_test(t, block_shape, paddings)

    def test_masked(self):
        t = tf.random.uniform([16, 20, 10])  # e.g. [batch, time, features]
        block_shape = [4]
        paddings = [[2, 2]]
        self._generic_masked_test(t, block_shape, paddings)

    @staticmethod
    def _generic_public_test(t, block_shape, paddings):
        actual = tf.space_to_batch(t, block_shape=block_shape, paddings=paddings)
        b = tfe.define_public_variable(t)
        final = tfe.space_to_batch(
            b, block_shape=block_shape, paddings=paddings
        ).to_native()
        np.testing.assert_array_almost_equal(final, actual, decimal=3)

    @staticmethod
    def _generic_private_test(t, block_shape, paddings):
        actual = tf.space_to_batch(t, block_shape=block_shape, paddings=paddings)
        b = tfe.define_private_variable(t)
        out = tfe.space_to_batch(b, block_shape=block_shape, paddings=paddings)
        final = out.reveal().to_native()
        np.testing.assert_array_almost_equal(final, actual, decimal=3)

    @staticmethod
    def _generic_masked_test(t, block_shape, paddings):
        actual = tf.space_to_batch(t, block_shape=block_shape, paddings=paddings)
        b = tfe.mask(tfe.define_private_variable(t))
        out = tfe.space_to_batch(b, block_shape=block_shape, paddings=paddings)
        final = out.reveal().to_native()
        np.testing.assert_array_almost_equal(final, actual, decimal=3)


class Testconcat(unittest.TestCase):
    def test_concat(self):

        t1 = [[1, 2, 3], [4, 5, 6]]
        t2 = [[7, 8, 9], [10, 11, 12]]
        actual = tf.concat([t1, t2], 0)
        x = tfe.define_private_variable(np.array(t1))
        y = tfe.define_private_variable(np.array(t2))
        out = tfe.concat([x, y], 0)
        final = out.reveal().to_native()
        np.testing.assert_array_equal(final, actual)

    @unittest.skipIf(isinstance(tfe.get_protocol(), ABY3), "ABY3 not support")
    def test_masked_concat(self):

        t1 = [[1, 2, 3], [4, 5, 6]]
        t2 = [[7, 8, 9], [10, 11, 12]]
        actual = tf.concat([t1, t2], 0)

        x = tfe.mask(tfe.define_private_variable(np.array(t1)))
        y = tfe.mask(tfe.define_private_variable(np.array(t2)))

        out = tfe.concat([x, y], 0)
        final = out.unmasked.reveal().to_native()

        np.testing.assert_array_equal(final, actual)


class TestConv2D(unittest.TestCase):
    def test_forward(self) -> None:
        # input
        batch_size, channels_in, channels_out = 32, 3, 64
        img_height, img_width = 28, 28
        input_shape = (batch_size, channels_in, img_height, img_width)
        input_conv = np.random.normal(size=input_shape).astype(np.float32)

        # filters
        strides = (2, 2)
        filter_shape = (2, 2)
        filter_values = np.random.normal(size=[2, 2, 3, 64])

        # convolution
        conv_input = tfe.define_private_variable(input_conv)
        conv_layer = tfe.keras.layers.Conv2D(
            channels_out,
            filter_shape,
            strides=strides,
            use_bias=False,
            data_format="channels_first",
        )
        conv_layer.build(input_shape)
        conv_layer.set_weights([filter_values])
        conv_out = conv_layer.call(conv_input)
        out = conv_out.reveal().to_native()

        x = tf.Variable(input_conv, dtype=tf.float32)
        x_nhwc = tf.transpose(x, (0, 2, 3, 1))
        # convolution Tensorflow
        filters_tf = tf.Variable(filter_values, dtype=tf.float32)
        conv_out_tf = tf.nn.conv2d(
            x_nhwc,
            filters_tf,
            strides=strides,
            padding="SAME",
        )
        out_tensorflow = tf.transpose(conv_out_tf, (0, 3, 1, 2))
        np.testing.assert_allclose(out, out_tensorflow, atol=0.01)

    def test_forward_bias(self) -> None:
        # input
        batch_size, channels_in, channels_out = 32, 3, 64
        img_height, img_width = 28, 28
        input_shape = (batch_size, channels_in, img_height, img_width)
        input_conv = np.random.normal(size=input_shape).astype(np.float32)

        # filters
        strides = (2, 2)
        filter_shape = (2, 2)
        filter_values = np.random.normal(size=[2, 2, 3, 64])

        # convolution
        conv_input = tfe.define_private_variable(input_conv)
        conv_layer = tfe.keras.layers.Conv2D(
            channels_out, filter_shape, strides=strides, data_format="channels_first"
        )
        conv_layer.build(input_shape)

        output_shape = conv_layer.compute_output_shape(input_shape)
        bias = np.random.uniform(size=[1, output_shape[1], 1, 1])
        conv_layer.set_weights([filter_values, bias])

        conv_out = conv_layer.call(conv_input)
        out = conv_out.reveal().to_native()

        x = tf.Variable(input_conv, dtype=tf.float32)
        x_nhwc = tf.transpose(x, (0, 2, 3, 1))

        # convolution Tensorflow
        filters_tf = tf.Variable(filter_values, dtype=tf.float32)

        conv_out_tf = tf.nn.conv2d(
            x_nhwc,
            filters_tf,
            strides=strides,
            padding="SAME",
        )
        out_tensorflow = tf.transpose(conv_out_tf, (0, 3, 1, 2))
        out_tensorflow += bias

        np.testing.assert_allclose(out, out_tensorflow, atol=0.01)

    def test_backward(self):
        pass


class TestMatMul(unittest.TestCase):
    def test_matmul(self) -> None:
        input_shape = [4, 5]
        x_in = np.random.normal(size=input_shape)
        filter_shape = [5, 4]
        filter_values = np.random.normal(size=filter_shape)
        input_input = tfe.define_private_variable(x_in)
        filter_filter = tfe.define_private_variable(filter_values)
        out = tfe.matmul(input_input, filter_filter)
        out = out.reveal().to_native()

        x = tf.Variable(x_in, dtype=tf.float32)
        filters_tf = tf.Variable(filter_values, dtype=tf.float32)
        out = tf.matmul(x, filters_tf)

        np.testing.assert_array_almost_equal(out, out, decimal=2)

    def test_big_middle_matmul(self) -> None:
        input_shape = [64, 4500]
        x_in = np.random.normal(size=input_shape)
        filter_shape = [4500, 64]
        filter_values = np.random.normal(size=filter_shape)
        input_input = tfe.define_private_variable(x_in)
        filter_filter = tfe.define_private_variable(filter_values)
        out = tfe.matmul(input_input, filter_filter)
        out = out.reveal().to_native()

        x = tf.Variable(x_in, dtype=tf.float32)
        filters_tf = tf.Variable(filter_values, dtype=tf.float32)
        out = tf.matmul(x, filters_tf)

        np.testing.assert_allclose(out, out, atol=0.1)


class TestNegative(unittest.TestCase):
    def test_negative(self):
        input_shape = [2, 2]
        input_neg = np.ones(input_shape)

        # reshape
        neg_input = tfe.define_private_variable(input_neg)
        neg_out = tfe.negative(neg_input)
        out = neg_out.reveal().to_native()

        x = tf.Variable(input_neg, dtype=tf.float32)
        neg_out_tf = tf.negative(x)

        assert np.isclose(out, neg_out_tf, atol=0.6).all()


class TestSqrt(unittest.TestCase):
    def test_sqrt(self):
        input_shape = [2, 2]
        input_sqrt = np.ones(input_shape)

        # reshape
        sqrt_input = tfe.define_public_variable(input_sqrt)
        sqrt_out = tfe.sqrt(sqrt_input).to_native()

        x = tf.Variable(input_sqrt, dtype=tf.float32)
        sqrt_out_tf = tf.math.sqrt(x)
        assert np.isclose(sqrt_out, sqrt_out_tf, atol=0.6).all()


class TestPad(unittest.TestCase):
    def test_pad(self):
        x_in = np.array([[1, 2, 3], [4, 5, 6]])
        input_input = tfe.define_private_variable(x_in)
        paddings = [[2, 2], [3, 4]]
        out = tfe.pad(input_input, paddings)
        out = out.reveal().to_native()

        out_tensorflow = tf.pad(tf.convert_to_tensor(x_in), paddings)
        np.testing.assert_allclose(out.numpy(), out_tensorflow.numpy(), atol=0.01)


@unittest.skipIf(
    isinstance(tfe.get_protocol(), Pond)
    and not isinstance(tfe.get_protocol(), SecureNN),
    "Pond not support",
)
class TestReduceMax(unittest.TestCase):
    def test_reduce_max_1d(self):

        t = np.array([1, 2, 3, 4]).astype(float)
        expected = tf.reduce_max(t)

        b = tfe.define_private_variable(tf.constant(t))
        out_tfe = tfe.reduce_max(b)
        for _ in range(2):
            actual = out_tfe.reveal().to_native()

        np.testing.assert_array_equal(actual, expected)

    def test_reduce_max_2d_axis0(self):

        t = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape((2, 4)).astype(float)
        expected = tf.reduce_max(t, axis=0)

        b = tfe.define_private_variable(tf.constant(t))
        out_tfe = tfe.reduce_max(b, axis=0)
        for _ in range(2):
            actual = out_tfe.reveal().to_native()

        np.testing.assert_array_equal(actual, expected)

    def test_reduce_max_2d_axis1(self):

        t = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape((2, 4)).astype(float)
        expected = tf.reduce_max(t, axis=1)

        b = tfe.define_private_variable(tf.constant(t))
        out_tfe = tfe.reduce_max(b, axis=1)
        for _ in range(2):
            actual = out_tfe.reveal().to_native()

        np.testing.assert_array_equal(actual, expected)

    def test_reduce_max_3d_axis0(self):

        t = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape((2, 2, 2))
        expected = tf.reduce_max(t, axis=0)

        b = tfe.define_private_variable(tf.constant(t))
        out_tfe = tfe.reduce_max(b, axis=0)
        for _ in range(2):
            actual = out_tfe.reveal().to_native()

        np.testing.assert_array_equal(actual, expected)


class TestReduceSum(unittest.TestCase):
    def test_reduce_sum_1d(self):

        t = [1, 2]
        actual = tf.reduce_sum(t)

        b = tfe.define_private_variable(tf.constant(t))
        out = tfe.reduce_sum(b)
        final = out.reveal().to_native()

        np.testing.assert_array_equal(final, actual)

    def test_reduce_sum_2d(self):

        t = [[1, 2], [1, 3]]
        actual = tf.reduce_sum(t, axis=1)

        b = tfe.define_private_variable(tf.constant(t))
        out = tfe.reduce_sum(b, axis=1)
        final = out.reveal().to_native()

        np.testing.assert_array_equal(final, actual)

    def test_reduce_sum_huge_vector(self):

        t = [1] * 2**13
        actual = tf.reduce_sum(t)

        b = tfe.define_private_variable(tf.constant(t))
        out = tfe.reduce_sum(b)
        final = out.reveal().to_native()

        np.testing.assert_array_equal(final, actual)


class TestStack(unittest.TestCase):
    def test_stack(self):

        x = tf.constant([1, 4])
        y = tf.constant([2, 5])
        z = tf.constant([3, 6])
        actual = tf.stack([x, y, z])

        x = tfe.define_private_variable(np.array([1, 4]))
        y = tfe.define_private_variable(np.array([2, 5]))
        z = tfe.define_private_variable(np.array([3, 6]))

        out = tfe.stack((x, y, z), axis=0)
        final = out.reveal().to_native()

        np.testing.assert_array_equal(final, actual)


class TestStridedSlice(unittest.TestCase):
    def test_strided_slice(self):

        t = tf.constant(
            [
                [[1, 1, 1], [2, 2, 2]],
                [[3, 3, 3], [4, 4, 4]],
                [[5, 5, 5], [6, 6, 6]],
            ]
        )
        actual = tf.strided_slice(t, [1, 0, 0], [2, 1, 3], [1, 1, 1])

        x = np.array(
            [
                [[1, 1, 1], [2, 2, 2]],
                [[3, 3, 3], [4, 4, 4]],
                [[5, 5, 5], [6, 6, 6]],
            ]
        )

        out = tfe.define_private_variable(x)
        out = tfe.strided_slice(out, [1, 0, 0], [2, 1, 3], [1, 1, 1])
        final = out.reveal().to_native()

        np.testing.assert_array_equal(final, actual)


if __name__ == "__main__":
    unittest.main()
