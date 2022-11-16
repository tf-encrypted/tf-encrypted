# pylint: disable=missing-docstring
import logging
import sys
import unittest
from typing import List

import numpy as np
import tensorflow as tf
import tf2onnx

import tf_encrypted as tfe
from tf_encrypted.convert import Converter
from tf_encrypted.protocol.aby3 import ABY3


class TestConvertNodes(unittest.TestCase):
    def setUp(self):
        tfe.get_config().set_debug_mode(True)
        prot = ABY3()
        tfe.set_protocol(prot)
        self.previous_logging_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)

    def tearDown(self):
        logging.getLogger().setLevel(self.previous_logging_level)

    def _build_test(self, op_name, x, decimals=3, backward=True):
        model_builder = globals()["{}_model".format(op_name)]
        tf_model = model_builder(x)
        with tf.GradientTape() as tape:
            tape.watch(x)
            actual = tf_model(x)
        tf_dy_dx = tape.gradient(actual, x)

        spec = []
        input_shapes = []
        for input in x:
            input_shapes.append(list(input.shape))
            spec.append(tf.TensorSpec(input.shape, tf.float32))
        onnx_model, _ = tf2onnx.convert.from_keras(tf_model, input_signature=spec)

        c = Converter(config=tfe.get_config())
        tfe_model = c.convert(
            onnx_model, input_shapes, model_provider="weights-provider"
        )
        x = tfe.define_private_input("prediction-client", lambda: x)

        output = tfe_model.predict(x, reveal=True)
        for o_i, a_i in zip(output, actual):
            np.testing.assert_array_almost_equal(o_i, a_i, decimal=decimals)

        if backward:
            optimizer = tfe.keras.optimizers.SGD(learning_rate=0.001)
            loss = tfe.keras.losses.MeanSquaredError()
            tfe_model.compile(optimizer, loss)
            output = tfe.define_private_variable(np.ones(output.shape))
            tfe_dy_dx = tfe_model.backward(output)
            assert len(tf_dy_dx) == len(tfe_dy_dx)
            for index, dy_dx in enumerate(tf_dy_dx):
                np.testing.assert_array_almost_equal(
                    dy_dx, tfe_dy_dx[index].reveal().to_native(), decimal=decimals
                )

    def test_add(self):
        test_input = [tf.random.uniform([28, 1]), tf.random.uniform([28, 1])]
        self._build_test("add", test_input)

    def test_sub(self):
        test_input = [tf.random.uniform([28, 1]), tf.random.uniform([28, 1])]
        self._build_test("sub", test_input)

    def test_mul(self):
        test_input = [tf.random.uniform([28, 1]), tf.random.uniform([28, 1])]
        self._build_test("mul", test_input)

    def test_neg(self):
        test_input = [tf.random.uniform([28, 1])]
        self._build_test("neg", test_input)

    def test_matmul(self):
        test_input = [tf.random.uniform([1, 28])]
        self._build_test("matmul", test_input)

    def test_gemm(self):
        test_input = [tf.random.uniform([28, 1])]
        self._build_test("gemm", test_input)

    def test_conv2d(self):
        test_input = [tf.random.uniform([1, 224, 224, 3])]
        self._build_test("conv2d", test_input)

    def test_depthwise_conv2d(self):
        test_input = [tf.random.uniform([1, 28, 28, 3])]
        self._build_test("depthwise_conv2d", test_input, backward=False)

    def test_batchnorm(self):
        test_input = [tf.random.uniform([1, 1, 28, 28])]
        self._build_test("batchnorm", test_input, backward=False)

    def test_avgpool(self):
        test_input = [tf.random.uniform([1, 28, 28, 3])]
        self._build_test("avgpool", test_input, decimals=2)

    def test_global_avgpool(self):
        test_input = [tf.random.uniform([1, 28, 28, 3])]
        self._build_test("global_avgpool", test_input, decimals=2)

    def test_relu(self):
        test_input = [tf.random.uniform([1, 28, 28, 3])]
        self._build_test("relu", test_input, decimals=2)

    def test_sigmiod(self):
        test_input = [tf.random.uniform([1, 28, 28, 3])]
        self._build_test("sigmoid", test_input, decimals=1)

    def test_softmax(self):
        test_input = [tf.random.uniform([1, 28, 28, 3])]
        self._build_test("softmax", test_input, decimals=2, backward=False)

    def test_maxpool(self):
        test_input = [tf.random.uniform([1, 28, 28, 3])]
        self._build_test("maxpool", test_input)

    def test_global_maxgpool(self):
        test_input = [tf.random.uniform([1, 4, 4, 9])]
        self._build_test("global_maxpool", test_input)

    def test_strided_slice(self):
        test_input = [tf.random.uniform([10, 10, 10])]
        self._build_test("strided_slice", test_input, backward=False)

    def test_slice(self):
        test_input = [tf.random.uniform([10, 10, 10])]
        self._build_test("slice", test_input, backward=False)

    def test_gather(self):
        test_input = [tf.random.uniform([10, 10, 10])]
        self._build_test("gather", test_input, backward=False)

    def test_split(self):
        test_input = [tf.random.uniform([1, 10, 10, 3])]
        self._build_test("split", test_input, backward=False)

    def test_split_edge_case(self):
        test_input = [tf.random.uniform([1, 10, 10, 4])]
        self._build_test("split_edge_case", test_input, backward=False)

    def test_split_v(self):
        test_input = [tf.random.uniform([1, 10, 10, 3])]
        self._build_test("split_v", test_input, backward=False)

    def test_concat(self):
        test_input = [
            tf.random.uniform([1, 10, 10, 3]),
            tf.random.uniform([1, 10, 10, 3]),
            tf.random.uniform([1, 10, 10, 3]),
        ]
        self._build_test("concat", test_input)

    def test_stack(self):
        test_input = [
            tf.random.uniform([1, 10, 10, 3]),
            tf.random.uniform([1, 10, 10, 3]),
            tf.random.uniform([1, 10, 10, 3]),
        ]
        self._build_test("stack", test_input)

    def test_reshape(self):
        test_input = [tf.random.uniform([1, 2, 3, 4])]
        self._build_test("reshape", test_input)

    def test_transpose(self):
        test_input = [tf.random.uniform([1, 2, 3, 4])]
        self._build_test("transpose", test_input)

    def test_expand_dims(self):
        test_input = [tf.random.uniform([2, 3, 4])]
        self._build_test("expand_dims", test_input)

    def test_squeeze(self):
        test_input = [tf.random.uniform([1, 2, 3, 1])]
        self._build_test("squeeze", test_input)

    def test_pad(self):
        test_input = [tf.ones([2, 2, 2, 2])]
        self._build_test("pad", test_input)

    def test_multilayer(self):
        test_input = [tf.random.uniform([1, 8, 8, 1])]
        self._build_test("multilayer", test_input)


def add_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])
    y = tf.keras.layers.Input(shape=inputs[1].shape[1:])
    res = tf.keras.layers.Add()([x, y])
    model = tf.keras.models.Model(inputs=[x, y], outputs=res)
    return model


def sub_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])
    y = tf.keras.layers.Input(shape=inputs[1].shape[1:])
    res = tf.keras.layers.Subtract()([x, y])
    model = tf.keras.models.Model(inputs=[x, y], outputs=res)
    return model


def mul_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])
    y = tf.keras.layers.Input(shape=inputs[1].shape[1:])
    res = tf.keras.layers.Multiply()([x, y])
    model = tf.keras.models.Model(inputs=[x, y], outputs=res)
    return model


def neg_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])
    res = -x
    model = tf.keras.models.Model(inputs=x, outputs=res)
    return model


def matmul_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])

    dense_layer = tf.keras.layers.Dense(28, use_bias=False)
    dense_layer.build(inputs[0].shape)
    weights = dense_layer.weights
    kernel = tf.random.uniform(weights[0].shape, dtype=np.float32)
    dense_layer.set_weights([kernel])

    res = dense_layer(x)
    model = tf.keras.models.Model(inputs=x, outputs=res)
    return model


def gemm_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])

    dense_layer = tf.keras.layers.Dense(28, use_bias=True)
    dense_layer.build(inputs[0].shape)
    weights = dense_layer.weights
    kernel = tf.random.uniform(weights[0].shape, dtype=np.float32)
    bias = tf.random.uniform(weights[1].shape, dtype=np.float32)
    dense_layer.set_weights([kernel, bias])

    res = dense_layer(x)
    model = tf.keras.models.Model(inputs=x, outputs=res)
    return model


def conv2d_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])

    conv2d_layer = tf.keras.layers.Conv2D(3, 3)
    conv2d_layer.build(inputs[0].shape)
    weights = conv2d_layer.weights
    kernel = tf.random.uniform(weights[0].shape, dtype=np.float32)
    bias = tf.random.uniform(weights[1].shape, dtype=np.float32)
    conv2d_layer.set_weights([kernel, bias])

    res = conv2d_layer(x)
    model = tf.keras.models.Model(inputs=x, outputs=res)
    return model


def depthwise_conv2d_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])

    depthwise_conv2d_layer = tf.keras.layers.DepthwiseConv2D(3, depth_multiplier=3)
    depthwise_conv2d_layer.build(inputs[0].shape)
    weights = depthwise_conv2d_layer.weights
    kernel = tf.random.uniform(weights[0].shape, dtype=np.float32)
    bias = tf.random.uniform(weights[1].shape, dtype=np.float32)
    depthwise_conv2d_layer.set_weights([kernel, bias])

    res = depthwise_conv2d_layer(x)
    model = tf.keras.models.Model(inputs=x, outputs=res)
    return model


def batchnorm_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])

    batch_norm_layer = tf.keras.layers.BatchNormalization()
    batch_norm_layer.build(inputs[0].shape)
    weights = batch_norm_layer.weights
    gamma = tf.random.uniform(weights[0].shape, dtype=np.float32)
    beta = tf.random.uniform(weights[1].shape, dtype=np.float32)
    mean = tf.random.uniform(weights[2].shape, dtype=np.float32)
    variance = tf.random.uniform(weights[3].shape, dtype=np.float32)
    batch_norm_layer.set_weights([gamma, beta, mean, variance])

    res = batch_norm_layer(x)
    model = tf.keras.models.Model(inputs=x, outputs=res)
    return model


def avgpool_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])
    res = tf.keras.layers.AveragePooling2D((2, 2), 2)(x)
    model = tf.keras.models.Model(inputs=x, outputs=res)
    return model


def global_avgpool_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])
    res = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.models.Model(inputs=x, outputs=res)
    return model


def relu_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])
    res = tf.keras.layers.ReLU()(x)
    model = tf.keras.models.Model(inputs=x, outputs=res)
    return model


def sigmoid_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])
    res = tf.keras.activations.sigmoid(x)
    model = tf.keras.models.Model(inputs=x, outputs=res)
    return model


def softmax_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])
    res = tf.keras.layers.Softmax()(x)
    model = tf.keras.models.Model(inputs=x, outputs=res)
    return model


def maxpool_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])
    res = tf.keras.layers.MaxPool2D(2, 2)(x)
    model = tf.keras.models.Model(inputs=x, outputs=res)
    return model


def global_maxpool_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])
    res = tf.keras.layers.GlobalMaxPool2D()(x)
    model = tf.keras.models.Model(inputs=x, outputs=res)
    return model


def strided_slice_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])
    start = [1, 2]
    end = [3, 5]
    strides = [1, 1]
    res = tf.strided_slice(x, start, end, strides)
    model = tf.keras.models.Model(inputs=x, outputs=res)
    return model


def slice_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])
    start = [1, 0, 0]
    size = [2, 4, 4]
    res = tf.slice(x, start, size)
    model = tf.keras.models.Model(inputs=x, outputs=res)
    return model


def gather_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])
    res = tf.gather(x, indices=[1, 3], axis=0)
    model = tf.keras.models.Model(inputs=x, outputs=res)
    return model


def split_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])
    res = tf.split(x, num_or_size_splits=3, axis=-1)
    model = tf.keras.models.Model(inputs=x, outputs=res)
    return model


def split_edge_case_model(inputs: List[tf.Tensor]) -> tf.Module:
    init = tf.keras.initializers.RandomNormal(seed=1)
    x = tf.keras.Input(shape=inputs[0].shape[1:])
    y1, y2 = tf.keras.layers.Lambda(
        lambda tensor: tf.split(tensor, num_or_size_splits=2, axis=-1)
    )(x)
    y = tf.keras.layers.Conv2D(
        2, 3, kernel_initializer=init, use_bias=True, padding="same"
    )(y2)
    y = tf.keras.layers.Concatenate(axis=-1)([y1, y])
    model = tf.keras.models.Model(inputs=x, outputs=y)
    return model


def split_v_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])
    res = tf.split(x, num_or_size_splits=[1, 1, 1], axis=-1)
    model = tf.keras.models.Model(inputs=x, outputs=res)
    return model


def concat_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])
    y = tf.keras.layers.Input(shape=inputs[1].shape[1:])
    z = tf.keras.layers.Input(shape=inputs[2].shape[1:])
    res = tf.concat([x, y, z], axis=-1)
    model = tf.keras.models.Model(inputs=[x, y, z], outputs=res)
    return model


def stack_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])
    y = tf.keras.layers.Input(shape=inputs[1].shape[1:])
    z = tf.keras.layers.Input(shape=inputs[2].shape[1:])
    res = tf.stack([x, y, z])
    model = tf.keras.models.Model(inputs=[x, y, z], outputs=res)
    return model


def reshape_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])
    last_size = 1
    for i in inputs[0].shape[1:]:
        last_size *= i
    res = tf.reshape(x, [-1, last_size])
    model = tf.keras.models.Model(inputs=x, outputs=res)
    return model


def transpose_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])
    res = tf.transpose(x, perm=(0, 3, 1, 2))
    model = tf.keras.models.Model(inputs=x, outputs=res)
    return model


def expand_dims_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])
    res = tf.expand_dims(x, axis=0)
    model = tf.keras.models.Model(inputs=x, outputs=res)
    return model


def squeeze_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])
    res = tf.squeeze(x, axis=[0, 3])
    model = tf.keras.models.Model(inputs=x, outputs=res)
    return model


def pad_model(inputs: List[tf.Tensor]) -> tf.Module:
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])
    res = tf.pad(
        x, paddings=tf.constant([[1, 2], [3, 4], [5, 6], [7, 8]]), mode="CONSTANT"
    )
    model = tf.keras.models.Model(inputs=x, outputs=res)
    return model


def multilayer_model(inputs: List[tf.Tensor]) -> tf.Module:
    init = tf.keras.initializers.RandomNormal(seed=1)
    x = tf.keras.layers.Input(shape=inputs[0].shape[1:])
    y = tf.keras.layers.Conv2D(2, 3, kernel_initializer=init)(x)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.MaxPooling2D(2)(y)
    y = tf.keras.layers.Flatten()(y)
    y = tf.keras.layers.Dense(2, kernel_initializer=init)(y)
    model = tf.keras.Model(x, y)
    return model


if __name__ == "__main__":

    if len(sys.argv) == 3:
        config_file = sys.argv[2]
        config = tfe.RemoteConfig.load(config_file)
        config.connect_servers()
        # backward tests only run in debug mode
        # debug mode also run faster
        config.set_debug_mode(True)
        tfe.set_config(config)
    else:
        config = tfe.LocalConfig(
            player_names=[
                "server0",
                "server1",
                "server2",
                "prediction-client",
                "weights-provider",
            ]
        )
        tfe.set_config(config)

    if len(sys.argv) >= 2:
        test = sys.argv[1]
        singletest = unittest.TestSuite()
        singletest.addTest(TestConvertNodes(test))
        unittest.TextTestRunner().run(singletest)
    else:
        unittest.main()
