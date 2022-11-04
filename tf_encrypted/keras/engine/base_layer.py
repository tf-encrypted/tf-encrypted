"""Includes base classes used by all layer types."""
import logging
from abc import ABC

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import generic_utils

import tf_encrypted as tfe
from tf_encrypted.keras.engine.base_layer_utils import unique_object_name
from tf_encrypted.protocol import TFEPrivateTensor
from tf_encrypted.protocol import TFEPrivateVariable

logger = logging.getLogger("tf_encrypted")


class Layer(ABC):
    """
    Base layer class.
    This is the class from which all layers inherit.
    A layer is a class implementing common neural networks operations, such
    as convolution, batch norm, etc. These operations require managing weights,
    losses, updates, and inter-layer connectivity.
    Users will just instantiate a layer and then treat it as a callable.
    We recommend that descendants of `Layer` implement the following methods:
    * `__init__()`: Save configuration in member variables
    * `build()`: Called once from `__call__`, when we know the shapes of inputs
        and `dtype`.
    * `call()`: Called in `__call__` after making sure `build()` has been called
        once. Should actually perform the logic of applying the layer to the
        input tensors (which should be passed in as the first argument).
    """

    def __init__(self, trainable=True, name=None, **kwargs):

        allowed_kwargs = {
            "input_shape",
            "batch_input_shape",
            "batch_size",
            "weights",
            "activity_regularizer",
            "dtype",
            "lazy_normalization",
        }
        # Validate optional keyword arguments.
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError("Keyword argument not understood:", kwarg)

        if "input_shape" in kwargs:
            logger.warning(
                "Currently input_shape argument semantics include the "
                "batch dimension. Please construct you model "
                "accordingly."
            )
            self._batch_input_shape = kwargs["input_shape"]
        if "batch_input_shape" in kwargs:
            self._batch_input_shape = kwargs["batch_input_shape"]

        self.lazy_normalization = False
        if "lazy_normalization" in kwargs:
            self.lazy_normalization = kwargs["lazy_normalization"]

        self.trainable = trainable
        self._init_set_name(name)
        self.built = False
        self.weights = []

    def build(self, input_shape):  # pylint: disable=unused-argument
        """Creates the variables of the layer (optional, for subclass implementers).
        This is a method that implementers of subclasses of `Layer`
        can override if they need a state-creation step in-between
        layer instantiation and layer call.
        This is typically used to create the weights of `Layer` subclasses.
        Arguments:
        input_shape: Instance of `TensorShape`, or list of instances of
            `TensorShape` if the layer expects a list of inputs
            (one instance per input).
        """
        self.built = True

    def call(self, inputs):
        """This is where the layer's logic lives.
        Arguments:
            inputs: Input tensor, or list/tuple of input tensors.
        Returns:
            A tensor or list/tuple of tensors.
        """
        return inputs

    def compute_output_shape(self, input_shape):
        """Returns the layer's output shape"""

    def __call__(self, inputs, *args, **kargs):
        """Wraps `call`, applying pre- and post-processing steps.
        Arguments:
        inputs: input tensor(s).
        *args: additional positional arguments to be passed to `self.call`.
        **kwargs: additional keyword arguments to be passed to `self.call`.
        Returns:
        Output tensor(s).
        """
        if not self.built:
            input_shapes = inputs.shape
            self.build(input_shapes)

            self.built = True

        with tf.name_scope(self._name):
            outputs = self.call(inputs, *args, **kargs)

        return outputs

    def add_weight(self, variable, make_private=True):
        if make_private:
            variable = tfe.define_private_variable(variable)
            self.weights.append(variable)
        else:
            variable = tfe.define_public_variable(variable)
            self.weights.append(variable)

        return variable

    def set_weights(self, weights):
        """Sets the weights of the layer.
        Arguments:
        weights: A list of Numpy arrays with shapes and types
            matching the output of layer.get_weights() or a list
            of private tensors
        """
        weights_types = (np.ndarray, TFEPrivateTensor)
        assert isinstance(weights[0], weights_types), type(weights[0])

        if isinstance(weights[0], np.ndarray):
            for index, weight in enumerate(weights):
                weights[index] = tfe.define_private_variable(weight)
        if isinstance(weights[0], TFEPrivateVariable):
            for index, weight in enumerate(weights):
                weights[index] = weights[index].read_value()
        for i, w in enumerate(self.weights):
            shape = w.shape.as_list()
            tfe.assign(w, weights[i].reshape(shape))

    @property
    def name(self):
        return self._name

    def _init_set_name(self, name, zero_based=True):
        if not name:
            self._name = unique_object_name(
                generic_utils.to_snake_case(self.__class__.__name__),
                zero_based=zero_based,
            )
        else:
            self._name = name
