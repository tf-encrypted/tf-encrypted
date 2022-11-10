"""Register nodes for the TF Encrypted Converter."""
from abc import ABC
from abc import abstractmethod
from copy import deepcopy
from functools import reduce
from typing import List

import numpy as np
import tensorflow as tf
from onnx.onnx_ml_pb2 import NodeProto

import tf_encrypted as tfe
from tf_encrypted.player import Player

from ..keras.layers import Activation
from ..keras.layers import AveragePooling2D
from ..keras.layers import BatchNormalization
from ..keras.layers import Conv2D
from ..keras.layers import Dense
from ..keras.layers import DepthwiseConv2D
from ..keras.layers import GlobalAveragePooling2D
from ..keras.layers import GlobalMaxPooling2D
from ..keras.layers import MaxPooling2D
from ..keras.layers import ReLU


class BaseNode(ABC):
    """
    An abstraction for node in an ONNX computation graph

    Args:
        node: ONNX node proto
        inputs: a node's all inputs.
            If a input is other node's output,
            it will be substituted with a tensor shape
        model_provider: The Player who will act as the model provider, or a string
            identifier for the Player.
    """

    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        self.node = node
        self.inputs = inputs
        self.model_provider = model_provider
        # node's weights
        self._weights = []
        # node's input shapes, only include input from other node's output
        self._input_shapes = []
        # node's output shapes
        self._output_shapes = []

    @property
    def weights(self):
        return self._weights

    @property
    def input_shape(self):
        return self._input_shapes

    @property
    def output_shape(self):
        return self._output_shapes

    @abstractmethod
    def forward(self, x):
        """
        Implementation of node's forward function

        Args:
            x: a list of TFETensor from other node's forward ouptut
        """
        pass

    @abstractmethod
    def backward(self, d_y):
        """
        Implementation of node's backward function

        Args:
            d_y: a list of TFETensor from other node's backard output
        """
        pass


class AddNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(AddNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])
        self._output_shapes.append(inputs[0])
        input_b = inputs[1]

        if isinstance(input_b, tf.Tensor):
            assert isinstance(inputs[1], tf.Tensor), "bias must be a tensor"
            self.bias = tfe.define_private_input(model_provider, lambda: inputs[1])
            self.bias = tfe.define_private_variable(self.bias)
            self._weights.append(self.bias)
        else:
            assert isinstance(inputs[1], list), "input shape must be a list"
            self.bias = None

    def forward(self, x):
        if self.bias is None:
            return [tfe.add(x[0], x[1])]
        else:
            return [tfe.add(x[0], self.bias)]

    def backward(self, d_y):
        d_y = d_y[0]
        grad_weights = []
        if self.bias is None:
            d_ys = [d_y, d_y]
        else:
            batch_size = d_y.shape[0]
            d_bias = d_y.reduce_sum(axis=0)
            d_bias = d_bias / batch_size
            grad_weights.append(d_bias)
            d_ys = [d_y]
        return grad_weights, d_ys


class SubNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(SubNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        assert isinstance(inputs[1], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])
        self._output_shapes.append(inputs[0])

    def forward(self, x):
        return [tfe.sub(x[0], x[1])]

    def backward(self, d_y):
        d_y = d_y[0]
        return [], [d_y, tfe.negative(d_y)]


class MulNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(MulNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        assert isinstance(inputs[1], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])
        self._output_shapes.append(inputs[0])

        self.a, self.b = None, None

    def forward(self, x):
        self.a, self.b = x[0], x[1]
        return [tfe.mul(x[0], x[1])]

    def backward(self, d_y):
        d_y = d_y[0]
        d_a = tfe.mul(self.b, d_y)
        d_b = tfe.mul(self.a, d_y)
        return [], [d_a, d_b]


class NegativeNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(NegativeNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])
        self._output_shapes.append(inputs[0])

    def forward(self, x):
        return [tfe.negative(x[0])]

    def backward(self, d_y):
        return [], [tfe.negative(d_y[0])]


class MatmulNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(MatmulNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])

        assert isinstance(inputs[1], tf.Tensor), "kernel must be a tensor"
        kernel = tfe.define_private_input(model_provider, lambda: inputs[1])
        weights = []
        weights.append(kernel)
        if len(inputs) == 3:
            use_bias = True
            assert isinstance(inputs[2], tf.Tensor), "bias must be a tensor"
            bias = tfe.define_private_input(model_provider, lambda: inputs[2])
            weights.append(bias)
        else:
            use_bias = False
            bias = None

        self.layer = Dense(kernel.shape[1], use_bias=use_bias)
        self.layer.build(self._input_shapes[0])
        self._output_shapes.append(
            self.layer.compute_output_shape(self._input_shapes[0])
        )
        self.layer.set_weights(weights)
        self._weights = self.layer.weights

    def forward(self, x):
        return [self.layer.call(x[0])]

    def backward(self, d_y):
        grad_weights, d_y = self.layer.backward(d_y[0])
        return grad_weights, [d_y]


class GemmNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(GemmNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])

        assert isinstance(inputs[1], tf.Tensor), "kernel must be a tensor"
        kernel = tfe.define_private_input(model_provider, lambda: inputs[1])
        assert isinstance(inputs[2], tf.Tensor), "bias must be a tensor"
        bias = tfe.define_private_input(model_provider, lambda: inputs[2])
        if node.attribute[1].i:
            kernel = kernel.transpose()
        weights = []
        weights.append(kernel)
        weights.append(bias)

        self.transpose_x = False
        if node.attribute[0].i:
            self.transpose_x = True

        self.layer = Dense(kernel.shape[1], use_bias=True)
        self.layer.build(self._input_shapes[0])
        self._output_shapes.append(
            self.layer.compute_output_shape(self._input_shapes[0])
        )
        self.layer.set_weights(weights)
        self._weights = self.layer.weights

    def forward(self, x):
        if self.transpose_x:
            x = x[0].transpose()
        else:
            x = x[0]
        return [self.layer.call(x)]

    def backward(self, d_y):
        grad_weights, d_y = self.layer.backward(d_y[0])
        if self.transpose_x:
            d_y = d_y.transpose()
        return grad_weights, [d_y]


class Conv2dNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(Conv2dNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])
        padded_shape = deepcopy(inputs[0])

        assert isinstance(inputs[1], tf.Tensor), "kernel must be a tensor"
        kernel = tfe.define_private_input(model_provider, lambda: inputs[1])
        kernel = kernel.transpose(perm=[2, 3, 1, 0])
        weights = []
        weights.append(kernel)
        if len(inputs) == 3:
            use_bias = True
            assert isinstance(inputs[2], tf.Tensor), "bias must be a tensor"
            bias = tfe.define_private_input(model_provider, lambda: inputs[2])
            for ax in [0, -1, -1]:
                bias = bias.expand_dims(axis=ax)
            weights.append(bias)
        else:
            use_bias = False
            bias = None

        attributes = {}
        for attr in node.attribute:
            attributes[attr.name] = attr

        strides = list(attributes["strides"].ints)
        group = attributes["group"].i
        try:
            self.padding = list(attributes["pads"].ints)
            padded_shape[2] += self.padding[0] + self.padding[2]
            padded_shape[3] += self.padding[1] + self.padding[3]
        except KeyError:
            self.padding = None

        if group == 1:
            self.layer = Conv2D(
                kernel.shape[3],
                kernel.shape[0:2],
                strides=strides,
                padding="VALID",
                data_format="channels_first",
                use_bias=use_bias,
            )
        else:
            self.layer = DepthwiseConv2D(
                kernel.shape[0:2],
                strides=strides,
                depth_multiplier=kernel.shape[3] // inputs[0][1],
                padding="VALID",
                data_format="channels_first",
                use_bias=use_bias,
            )
        self.layer.build(padded_shape)
        self.layer.set_weights(weights)
        self._weights = self.layer.weights
        self._output_shapes.append(self.layer.compute_output_shape(padded_shape))

    def forward(self, x):
        x = x[0]
        if self.padding is not None:
            x = tfe.pad(
                x,
                [
                    [0, 0],
                    [0, 0],
                    [self.padding[2], self.padding[0]],
                    [self.padding[1], self.padding[3]],
                ],
            )
        return [self.layer.call(x)]

    def backward(self, d_y):
        d_y = d_y[0]
        grad_weights, d_y = self.layer.backward(d_y)
        dy_shape = d_y.shape
        if self.padding is not None:
            d_y = d_y[
                :,
                :,
                self.padding[2] : dy_shape[2] - self.padding[0],
                self.padding[1] : dy_shape[3] - self.padding[3],
            ]
        return grad_weights, [d_y]


class BatchnormalizationNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(BatchnormalizationNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])

        epsilon = float(node.attribute[0].f)
        assert isinstance(inputs[1], tf.Tensor), "gamma must be a tensor"
        gamma = tf.keras.initializers.Constant(inputs[1])
        assert isinstance(inputs[2], tf.Tensor), "beta must be a tensor"
        beta = tf.keras.initializers.Constant(inputs[2])
        assert isinstance(inputs[3], tf.Tensor), "mean must be a tensor"
        moving_mean = tf.keras.initializers.Constant(inputs[3])
        assert isinstance(inputs[4], tf.Tensor), "variance must be a tensor"
        moving_variance = tf.keras.initializers.Constant(inputs[4])

        self.layer = BatchNormalization(
            axis=1,
            epsilon=epsilon,
            gamma_initializer=gamma,
            beta_initializer=beta,
            moving_mean_initializer=moving_mean,
            moving_variance_initializer=moving_variance,
        )
        self.layer.build(self._input_shapes[0])
        self._output_shapes.append(
            self.layer.compute_output_shape(self._input_shapes[0])
        )
        self._weights = self.layer.weights

    def forward(self, x):
        return [self.layer.call(x[0])]

    def backward(self, d_y):
        grad_weights, d_y = self.layer.backward(d_y[0])
        return grad_weights, [d_y]


class AvgpoolNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(AvgpoolNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])

        attributes = {}
        for attr in node.attribute:
            attributes[attr.name] = attr

        pool_shape = attributes["kernel_shape"].ints
        strides = attributes["strides"].ints

        self.layer = AveragePooling2D(pool_shape, strides, data_format="channels_first")
        self.layer.build(self._input_shapes[0])
        self._output_shapes.append(
            self.layer.compute_output_shape(self._input_shapes[0])
        )

    def forward(self, x):
        return [self.layer.call(x[0])]

    def backward(self, d_y):
        grad_weights, d_y = self.layer.backward(d_y[0])
        return grad_weights, [d_y]


class GlobalAvgpoolNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(GlobalAvgpoolNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])

        data_format = "channels_first"
        keepdims = True

        try:
            attributes = {}
            for attr in node.attribute:
                attributes[attr.name] = attr

            axes = attributes["axes"].ints
            keepdims = attributes["keepdims"].i == 1

            if axes[0] == 1 and axes[1] == 2:
                data_format = "channels_last"
            else:
                data_format = "channels_first"
        except:  # noqa:E722
            pass

        self.layer = GlobalAveragePooling2D(data_format=data_format, keepdims=keepdims)
        self.layer.build(self._input_shapes[0])
        self._output_shapes.append(
            self.layer.compute_output_shape(self._input_shapes[0])
        )

    def forward(self, x):
        return [self.layer.call(x[0])]

    def backward(self, d_y):
        grad_weights, d_y = self.layer.backward(d_y[0])
        return grad_weights, [d_y]


class ReluNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(ReluNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])

        self.layer = ReLU()
        self.layer.build(self._input_shapes[0])
        self._output_shapes.append(
            self.layer.compute_output_shape(self._input_shapes[0])
        )

    def forward(self, x):
        return [self.layer.call(x[0])]

    def backward(self, d_y):
        grad_weights, d_y = self.layer.backward(d_y[0])
        return grad_weights, [d_y]


class SigmoidNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(SigmoidNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])

        self.layer = Activation("sigmoid")
        self.layer.build(self._input_shapes[0])
        self._output_shapes.append(
            self.layer.compute_output_shape(self._input_shapes[0])
        )

    def forward(self, x):
        return [self.layer.call(x[0])]

    def backward(self, d_y):
        grad_weights, d_y = self.layer.backward(d_y[0])
        return grad_weights, [d_y]


class SoftmaxNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(SoftmaxNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])

        self.layer = Activation("softmax")
        self.layer.build(self._input_shapes[0])
        self._output_shapes.append(
            self.layer.compute_output_shape(self._input_shapes[0])
        )

    def forward(self, x):
        return [self.layer.call(x[0])]

    def backward(self, d_y):
        grad_weights, d_y = self.layer.backward(d_y[0])
        return grad_weights, [d_y]


class MaxPoolNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(MaxPoolNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])

        attributes = {}
        for attr in node.attribute:
            attributes[attr.name] = attr

        pool_shape = attributes["kernel_shape"].ints
        strides = attributes["strides"].ints

        self.layer = MaxPooling2D(pool_shape, strides, "VALID", "channels_first")
        self.layer.build(self._input_shapes[0])
        self._output_shapes.append(
            self.layer.compute_output_shape(self._input_shapes[0])
        )

    def forward(self, x):
        return [self.layer.call(x[0])]

    def backward(self, d_y):
        grad_weights, d_y = self.layer.backward(d_y[0])
        return grad_weights, [d_y]


class GlobalMaxPoolNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(GlobalMaxPoolNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])

        data_format = "channels_first"
        keepdims = False

        try:
            attributes = {}
            for attr in node.attribute:
                attributes[attr.name] = attr

            axes = attributes["axes"].ints
            keepdims = attributes["keepdims"].i == 1

            if axes[0] == 1 and axes[1] == 2:
                data_format = "channels_last"
            else:
                data_format = "channels_first"
        except:  # noqa:E722
            pass

        self.layer = GlobalMaxPooling2D(data_format=data_format, keepdims=keepdims)
        self.layer.build(self._input_shapes[0])
        self._output_shapes.append(
            self.layer.compute_output_shape(self._input_shapes[0])
        )

    def forward(self, x):
        return [self.layer.call(x[0])]

    def backward(self, d_y):
        grad_weights, d_y = self.layer.backward(d_y[0])
        return grad_weights, [d_y]


class StrideSliceNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(StrideSliceNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])

        self.begin = [0] * len(inputs[0])
        self.end = deepcopy(inputs[0])
        self.strides = [1] * len(inputs[0])
        if len(inputs) > 3:
            self.axes = inputs[3]
        else:
            self.axes = range(len(inputs[0]))

        if len(inputs) == 5:
            for index, a in enumerate(self.axes):
                self.strides[a] = inputs[4][index]

        for index, a in enumerate(self.axes):
            self.begin[a] = inputs[1][index]
            self.end[a] = inputs[2][index]

        output_shape = deepcopy(self.end)
        for index, dim in enumerate(output_shape):
            dim = dim - self.begin[index]
            dim = dim // self.strides[index]
        self._output_shapes.append(output_shape)

    def forward(self, x):
        return [
            tfe.strided_slice(
                x[0],
                self.begin,
                self.end,
                strides=self.strides,
            )
        ]

    def backward(self, d_y):
        # TODO: implement strideslice backward
        raise NotImplementedError()


class GatherNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(GatherNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])

        attributes = {}
        for attr in node.attribute:
            attributes[attr.name] = attr

        self.axis = attributes["axis"].i
        self.indices = inputs[1]

        output_shape = []
        for index, dim in enumerate(self._input_shapes[0]):
            if index == self.axis:
                output_shape.append(len(self.indices))
            else:
                output_shape.append(dim)
        self._output_shapes.append(output_shape)

    def forward(self, x):
        return [tfe.gather(x[0], self.indices, self.axis)]

    def backward(self, d_y):
        # TODO: implement strideslice backward
        raise NotImplementedError()


class SplitNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(SplitNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])

        self.axis = node.attribute[0].i

        if len(inputs) == 2:
            self.num_or_size_splits = inputs[1]
            for size_split in self.num_or_size_splits:
                output_shape = deepcopy(self._input_shapes[0])
                output_shape[self.axis] = size_split
                self._output_shapes.append(output_shape)
        else:
            self.num_or_size_splits = len(node.output)
            output_shape = deepcopy(self._input_shapes[0])
            output_shape[self.axis] /= self.num_or_size_splits
            self._output_shapes = [output_shape for i in range(self.num_or_size_splits)]

    def forward(self, x):
        return tfe.split(x[0], self.num_or_size_splits, self.axis)

    def backward(self, d_y):
        return [], [tfe.concat(d_y, self.axis)]


class ConcatNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(ConcatNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        self._input_shapes.extend(inputs)

        self.axis = node.attribute[0].i
        output_shape = deepcopy(inputs[0])
        output_shape[self.axis] = 0
        self.input_dims = []
        for input_shape in inputs:
            self.input_dims.append(input_shape[self.axis])
            output_shape[self.axis] += input_shape[self.axis]
        self._output_shapes.append(output_shape)

    def forward(self, x):
        return [tfe.concat(x, axis=self.axis)]

    def backward(self, d_y):
        d_y = d_y[0]
        return [], tfe.split(d_y, self.input_dims, axis=self.axis)


class StackNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(StackNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])

        self.axis = node.attribute[0].i
        output_shape = []
        for index, dim in enumerate(inputs[0]):
            if index == self.axis:
                output_shape.append(len(inputs))
            output_shape.append(dim)
        self._output_shapes.append(output_shape)

    def forward(self, x):
        return [tfe.stack(x, axis=self.axis)]

    def backward(self, d_y):
        d_y = d_y[0]
        d_ys = tfe.split(d_y, num_or_size_splits=1, axis=self.axis)
        for dy in d_ys:
            dy = tfe.squeeze(dy, axis=self.axis)
        return [], d_ys


class ReshapeNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(ReshapeNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])

        output_shape = list(inputs[1].numpy())
        self._output_shapes.append(output_shape)

    def forward(self, x):
        x = x[0]
        return [tfe.reshape(x, self._output_shapes[0])]

    def backward(self, d_y):
        d_y = d_y[0]
        return [], [tfe.reshape(d_y, self._input_shapes[0])]


class FlattenNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(FlattenNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])

        last_dim = reduce(lambda x, y: x * y, list(inputs[0][1:]))
        self._output_shapes.append([-1, last_dim])

    def forward(self, x):
        x = x[0]
        return [tfe.reshape(x, self._output_shapes[0])]

    def backward(self, d_y):
        d_y = d_y[0]
        return [], [tfe.reshape(d_y, self._input_shapes[0])]


class TransposeNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(TransposeNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])

        self.perm = np.array(node.attribute[0].ints)
        self.reversed_perm = np.argsort(self.perm)
        self._output_shapes.append(list(np.array(inputs[0])[self.perm]))

    def forward(self, x):
        return [tfe.transpose(x[0], perm=self.perm)]

    def backward(self, d_y):
        return [], [tfe.transpose(d_y[0], perm=self.reversed_perm)]


class ExpandDimsNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(ExpandDimsNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])

        self.axis = inputs[1].numpy().item()
        output_shape = []
        for index, dim in enumerate(inputs[0]):
            if index == self.axis:
                output_shape.append(1)
            output_shape.append(dim)
        self._output_shapes.append(deepcopy(output_shape))

    def forward(self, x):
        return [tfe.expand_dims(x[0], axis=self.axis)]

    def backward(self, d_y):
        return [], [tfe.reshape(d_y[0], self._input_shapes[0])]


class SqueezeNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(SqueezeNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])

        if len(inputs) == 2:
            self.axes = list(inputs[1].numpy())
        else:
            self.axes = None

        output_shape = []
        for index, dim in enumerate(self._input_shapes[0]):
            if dim == 1 and (self.axes is None or index in self.axes):
                continue
            output_shape.append(dim)
        self._output_shapes.append(output_shape)

    def forward(self, x):
        return [tfe.squeeze(x[0], axis=self.axes)]

    def backward(self, d_y):
        return [], [tfe.reshape(d_y[0], self._input_shapes[0])]


class PadNode(BaseNode):
    def __init__(
        self, node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
    ) -> None:
        super(PadNode, self).__init__(node, inputs, model_provider)
        assert isinstance(inputs[0], list), "input shape must be a list"
        self._input_shapes.append(inputs[0])

        p = list(inputs[1].numpy())
        dims = len(p) // 2
        self.paddings_lst = [[p[i], p[i + dims]] for i in range(0, dims)]

        output_shape = deepcopy(self._input_shapes[0])
        for i in range(len(output_shape)):
            output_shape[i] += self.paddings_lst[i][0] + self.paddings_lst[i][1]
        self._output_shapes.append(output_shape)

    def forward(self, x):
        return [tfe.pad(x[0], self.paddings_lst)]

    def backward(self, d_y):
        d_y = d_y[0]
        start = []
        end = []
        for i in range(len(self.paddings_lst)):
            start.append(self.paddings_lst[i][0])
        for i in range(len(self.paddings_lst)):
            end.append(d_y.shape[i] - self.paddings_lst[i][1])
        d_y = tfe.strided_slice(d_y, start, end)
        return [], [d_y]


# After implement a node, register the node here
nodes_dict = {
    "Add": AddNode,
    "Sub": SubNode,
    "Mul": MulNode,
    "Neg": NegativeNode,
    "MatMul": MatmulNode,
    "Gemm": GemmNode,
    "Conv": Conv2dNode,
    "BatchNormalization": BatchnormalizationNode,
    "AveragePool": AvgpoolNode,
    "ReduceMean": GlobalAvgpoolNode,
    "GlobalAveragePool": GlobalAvgpoolNode,
    "Relu": ReluNode,
    "Sigmoid": SigmoidNode,
    "Softmax": SoftmaxNode,
    "MaxPool": MaxPoolNode,
    "ReduceMax": GlobalMaxPoolNode,
    "GlobalMaxPool": GlobalMaxPoolNode,
    "Slice": StrideSliceNode,
    "Gather": GatherNode,
    "Split": SplitNode,
    "Concat": ConcatNode,
    "Stack": StackNode,
    "Reshape": ReshapeNode,
    "Flatten": FlattenNode,
    "Transpose": TransposeNode,
    "ExpandDims": ExpandDimsNode,
    "Unsqueeze": ExpandDimsNode,
    "Squeeze": SqueezeNode,
    "Pad": PadNode,
}


def build_node(
    node: NodeProto, inputs: List[tf.Tensor], model_provider: Player
) -> BaseNode:
    op_type = node.op_type
    return nodes_dict[op_type](node, inputs, model_provider)
