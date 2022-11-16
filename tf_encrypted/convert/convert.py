"""Module for automatically building TFE Model from
their corresponding TF Model.

See README.md for details on usage and extension."""
import functools
from typing import Any
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import onnx
import tensorflow as tf
import tf2onnx
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from onnx.onnx_ml_pb2 import ModelProto
from tf2onnx import optimizer

import tf_encrypted as tfe
from tf_encrypted.config import Config
from tf_encrypted.config import get_config
from tf_encrypted.keras import optimizers
from tf_encrypted.keras.models import BaseModel
from tf_encrypted.player import Player
from tf_encrypted.protocol import Protocol
from tf_encrypted.protocol.protocol import TFETensor

from .nodes import build_node


class Converter:
    """The TFE Converter.

    Args:
        config: `Config` to use when constructing the TFE graph.
            Defaults to the current global TFE config.
        protocol: `Protocol` to use when constructing the TFE graph.
            Defaults to the current global TFE protocol.
    """

    def __init__(
        self, config: Optional[Config] = None, protocol: Optional[Protocol] = None
    ) -> None:
        self.config = config if config is not None else get_config()
        if protocol is not None:
            tfe.set_protocol(protocol)
        self.initializers = {}
        self.trainable_weights = []
        self.node_op = {}

    def convert(
        self,
        model: Any,
        input_shapes: Union[List[int], List[List[int]]],
        model_provider: Optional[Union[str, Player]] = None,
    ) -> Any:
        """Convert a plaintext model to a TFE model.

        Args:
            model: palin tensorflow model, plain onnx model or model file path
            input_shapes: model's input shapes
            model_provider: The Player who will act as the model provider, or a string
            identifier for the Player.
        """
        if not isinstance(input_shapes[0], list):
            input_shapes = [input_shapes]

        if model_provider is None:
            model_provider = self.config.get_player("weights-provider")
        elif isinstance(model_provider, str):
            model_provider = self.config.get_player(model_provider)
        else:
            model_provider = model_provider
        assert isinstance(model_provider, Player)

        if isinstance(model, str):
            if model.endswith(".pb"):
                model = tf.keras.load_model(model)
                input_spec = []
                for input_shape in input_shapes:
                    input_spec.append(
                        tf.TensorSpec(input_shape, tf.float32, name="input")
                    )
                onnx_optimizers = optimizer._get_optimizers()
                # uncomment this line to disable batchnorm fuse
                # onnx_optimizers.pop("remove_back_to_back")
                model_proto, _ = tf2onnx.convert.from_keras(
                    model, input_signature=input_spec, optimizers=onnx_optimizers
                )
            elif model.endswith(".onnx"):
                model_proto = onnx.load(model)
            else:
                raise ValueError("Don't konw model type " + model)
        elif isinstance(model, tf.keras.Model):
            input_spec = []
            for input_shape in input_shapes:
                input_spec.append(tf.TensorSpec(input_shape, tf.float32, name="input"))
            onnx_optimizers = optimizer._get_optimizers()
            # uncomment this line to disable batchnorm fuse
            # onnx_optimizers.pop("remove_back_to_back")
            model_proto, _ = tf2onnx.convert.from_keras(
                model, input_signature=input_spec, optimizers=onnx_optimizers
            )
        elif isinstance(model, ModelProto):
            model_proto = model
        else:
            raise ValueError("Unknow model")
        assert isinstance(model_proto, ModelProto)

        class DefinedModel(BaseModel):
            """Model defined by a plain model."""

            def __init__(
                self, tfe_nodes, forward_func, backward_builder=None, name=None
            ):
                super(DefinedModel, self).__init__(name=name)
                self.tfe_nodes = tfe_nodes
                self.forward_func = forward_func
                self.weights = []
                for tfe_node in self.tfe_nodes.values():
                    self.weights.append(tfe_node.weights)
                self.backward_func = None
                self.backward_builder = backward_builder

            def call(self, inputs):
                if isinstance(inputs, TFETensor):
                    inputs = [inputs]
                assert isinstance(inputs[0], TFETensor)
                res = self.forward_func(inputs)
                if len(res) == 1:
                    res = res[0]
                return res

            def backward(self, d_y):
                if isinstance(d_y, TFETensor):
                    d_y = [d_y]
                assert isinstance(d_y[0], TFETensor)
                if self.backward_func is None:
                    raise NotImplementedError(
                        "Please compile this model before backward!"
                    )
                return self.backward_func(d_y)

            def compile(self, optimizer, loss):
                if self.backward_builder is None:
                    raise NotImplementedError("Backward is not support")
                self._optimizer = optimizers.get(optimizer)
                self._loss = loss
                assert self._optimizer is not None, "An optimizer must be specified."
                assert self._loss is not None, "A loss must be specified."

                self._optimizer.compile(self.weights)
                self.backward_func = self.backward_builder(optimizer=self._optimizer)

        for input_shape in input_shapes:
            input_shape[0] = -1
        initializers = self._build_initializer(model_proto)
        tfe_nodes, nodes_output = self._build_node(
            input_shapes, model_proto, initializers, model_provider
        )
        forward_func = self._build_forward(model_proto, tfe_nodes)
        backward_builder = functools.partial(
            self._build_backward,
            model_proto=model_proto,
            tfe_nodes=tfe_nodes,
            nodes_output=nodes_output,
        )
        model = DefinedModel(tfe_nodes, forward_func, backward_builder)
        return model

    def _build_initializer(self, model_proto) -> None:
        initializers = {}

        for initializer in model_proto.graph.initializer:
            name = initializer.name
            shape = initializer.dims
            data_type = TENSOR_TYPE_TO_NP_TYPE[initializer.data_type]
            data = np.frombuffer(initializer.raw_data, dtype=data_type)
            data = data.reshape(shape)
            data = tf.convert_to_tensor(data)
            initializers[name] = data
        return initializers

    def _build_node(
        self, model_input_shapes, model_proto, model_initializers, model_providers
    ):
        tfe_nodes = {}
        input_shapes = {}
        for index, input in enumerate(model_proto.graph.input):
            input_shapes[input.name] = model_input_shapes[index]

        for node in model_proto.graph.node:
            inputs = []
            for input in node.input:
                if input in model_initializers.keys():
                    inputs.append(model_initializers[input])
                elif input in input_shapes.keys():
                    inputs.append(input_shapes[input])
                else:
                    raise KeyError(
                        "Node " + node.name + " input: " + input + " doesn't found!"
                    )
            tfe_node = build_node(node, inputs, model_providers)
            tfe_nodes[node.name] = tfe_node
            for index, output in enumerate(node.output):
                input_shapes[output] = tfe_node.output_shape[index]

        return tfe_nodes, input_shapes.keys()

    def _build_forward(self, model_proto, tfe_nodes):
        nodes = model_proto.graph.node

        def forward_function(x):
            node_outputs = {}
            for index, input in enumerate(model_proto.graph.input):
                node_outputs[input.name] = x[index]
            for node in nodes:
                inputs = []
                for input in node.input:
                    if input in node_outputs.keys():
                        inputs.append(node_outputs[input])
                with tf.name_scope(node.name + "/forward"):
                    res = tfe_nodes[node.name].forward(inputs)
                for i, output in enumerate(node.output):
                    node_outputs[output] = res[i]
            res = []
            for output in model_proto.graph.output:
                res.append(node_outputs[output.name])
            return res

        return forward_function

    def _build_backward(self, model_proto, tfe_nodes, nodes_output, optimizer):
        nodes = model_proto.graph.node
        graph_input = model_proto.graph.input

        def backward_function(d_y):
            dy_dict = {}
            for index, output in enumerate(model_proto.graph.output):
                dy_dict[output.name] = d_y[index]
            for node in reversed(nodes):
                d_y = []
                tfe_node = tfe_nodes[node.name]
                for output in node.output:
                    d_y.append(dy_dict[output])

                with tf.name_scope(node.name + "/backward"):
                    grad_weights, d_y = tfe_node.backward(d_y)
                    optimizer.apply_gradients(tfe_node.weights, grad_weights)

                index = 0
                for input in node.input:
                    if input in nodes_output:
                        dy_dict[input] = d_y[index]
                        index += 1

            res_dy = []
            for input in graph_input:
                res_dy.append(dy_dict[input.name])
            return res_dy

        return backward_function
