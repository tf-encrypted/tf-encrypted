from __future__ import absolute_import
from typing import Tuple, List, Union, Optional, Any, NewType, Callable
import abc
import sys
import logging
from math import log2, ceil
from random import random
from functools import reduce

import numpy as np
import tensorflow as tf

from ..tensor.helpers import inverse
from ..tensor.factory import (
    AbstractFactory,
    AbstractTensor,
    AbstractConstant,
    AbstractVariable,
    AbstractPlaceholder,
)
from ..tensor.fixed import FixedpointConfig, _validate_fixedpoint_config
from ..tensor import int100factory, fixed100
from ..tensor import int64factory, fixed64
from ..types import Slice, Ellipse
from ..player import Player
from ..config import get_config, tensorflow_supports_int64
from .protocol import Protocol, global_cache_updaters, memoize, nodes


TFEData = Union[np.ndarray, tf.Tensor]
TFEVariable = Union["PondPublicVariable", "PondPrivateVariable", tf.Variable]
TFEPublicTensor = NewType("TFEPublicTensor", "PondPublicTensor")
TFETensor = Union[TFEPublicTensor, "PondPrivateTensor", "PondMaskedTensor"]
TFEInputter = Callable[[], Union[List[tf.Tensor], tf.Tensor]]
TF_INT_TYPES = [tf.int8, tf.int16, tf.int32, tf.int64]

_initializers = list()
_thismodule = sys.modules[__name__]


class Pond(Protocol):
    """
    Pond(server_0, server_1, crypto_producer, tensor_factory, fixedpoint_config)

    Pond is similar to SPDZ except it has been vectorized plus a few more optimizations.

    Pond works with 2 parties for computation and one crypto producer for triples.

    :param Player server_0: The "alice" of MPC.
    :param Player server_1: The "bob" of MPC.
    :param Player crypto_producer: The host to act as the crypto producer.  In `Pond` this party is
        responsible for producing triples to aid in computation.
    :param AbstractFactory tensor_factory: Which backing type of tensor you would like to use, e.g. `int100` or `int64`
    """  # noqa:E501

    def __init__(
        self,
        server_0: Optional[Player] = None,
        server_1: Optional[Player] = None,
        crypto_producer: Optional[Player] = None,
        tensor_factory: Optional[AbstractFactory] = None,
        fixedpoint_config: Optional[FixedpointConfig] = None,
    ) -> None:
        self.server_0 = server_0 or get_config().get_player("server0")
        self.server_1 = server_1 or get_config().get_player("server1")
        crypto_producer = crypto_producer or get_config().get_player("server2")
        crypto_producer = crypto_producer or get_config().get_player("crypto-producer")
        self.crypto_producer = crypto_producer

        if tensor_factory is None:
            if tensorflow_supports_int64():
                tensor_factory = int64factory
            else:
                logging.warning(
                    "Falling back to using int100 tensors due to lack of int64 support. "
                    "Performance may be improved by installing a version of TensorFlow "
                    "supporting this (1.13+ or custom build).")
                tensor_factory = int100factory

        if fixedpoint_config is None:
            if tensor_factory is int64factory:
                fixedpoint_config = fixed64
            elif tensor_factory is int100factory:
                fixedpoint_config = fixed100
            else:
                raise ValueError(
                    "Don't know how to pick fixedpoint configuration for tensor type {}".format(
                        tensor_factory
                    )
                )

        _validate_fixedpoint_config(fixedpoint_config, tensor_factory)
        self.fixedpoint_config = fixedpoint_config
        self.tensor_factory = tensor_factory

    def define_constant(
        self,
        value: np.ndarray,
        apply_scaling: bool = True,
        name: Optional[str] = None,
        factory: Optional[AbstractFactory] = None,
    ) -> "PondConstant":
        """
        define_constant(value, apply_scaling, name, factory) -> PondConstant

        Define a constant to use in computation.

        .. code-block:: python

            x = prot.define_constant(np.array([1,2,3,4]), apply_scaling=False)

        :See: tf.constant

        :param np.ndarray value: The value to define as a constant.
        :param bool apply_scaling: Whether or not to scale the value.
        :param str name: What name to give to this node in the graph.
        :param AbstractFactory factory: Which tensor type to represent this value with.
        """
        assert isinstance(value, np.ndarray), type(value)

        factory = factory or self.tensor_factory

        v = self._encode(value, apply_scaling)

        with tf.name_scope("constant{}".format("-" + name if name else "")):

            with tf.device(self.server_0.device_name):
                x_on_0 = factory.constant(v)

            with tf.device(self.server_1.device_name):
                x_on_1 = factory.constant(v)

        return PondConstant(self, x_on_0, x_on_1, apply_scaling)

    def define_public_placeholder(
        self,
        shape,
        apply_scaling: bool = True,
        name: Optional[str] = None,
        factory: Optional[AbstractFactory] = None,
    ) -> "PondPublicPlaceholder":
        """
        define_public_placeholder(shape, apply_scaling, name, factory) -> PondPublicPlaceholder

        Define a `public` placeholder to use in computation.  This will be known to both parties.

        .. code-block:: python

            x = prot.define_public_placeholder(shape=(1024, 1024))

        :See: tf.placeholder

        :param List[int] shape: The shape of the placeholder.
        :param bool apply_scaling: Whether or not to scale the value.
        :param str name: What name to give to this node in the graph.
        :param AbstractFactory factory: Which tensor type to represent this value with.
        """

        factory = factory or self.tensor_factory

        with tf.name_scope("public-placeholder{}".format("-" + name if name else "")):

            with tf.device(self.server_0.device_name):
                x_on_0 = factory.placeholder(shape)

            with tf.device(self.server_1.device_name):
                x_on_1 = factory.placeholder(shape)

        return PondPublicPlaceholder(self, x_on_0, x_on_1, apply_scaling)

    def define_private_placeholder(
        self,
        shape,
        apply_scaling: bool = True,
        name: Optional[str] = None,
        factory: Optional[AbstractFactory] = None,
    ) -> "PondPrivatePlaceholder":
        """
        define_private_placeholder(shape, apply_scaling, name, factory) -> PondPrivatePlaceholder

        Define a `private` placeholder to use in computation.  This will only be known by the party
        that defines it.

        .. code-block:: python

            x = prot.define_private_placeholder(shape=(1024, 1024))

        :See: tf.placeholder

        :param List[int] shape: The shape of the placeholder.
        :param bool apply_scaling: Whether or not to scale the value.
        :param str name: What name to give to this node in the graph.
        :param AbstractFactory factory: Which tensor type to represent this value with.
        """

        factory = factory or self.tensor_factory

        pl = factory.placeholder(shape)

        with tf.name_scope("private-placeholder{}".format("-" + name if name else "")):
            x0, x1 = self._share(pl)

        return PondPrivatePlaceholder(self, pl, x0, x1, apply_scaling)

    def define_public_variable(
        self,
        initial_value,
        apply_scaling: bool = True,
        name: Optional[str] = None,
        factory: Optional[AbstractFactory] = None,
    ) -> "PondPublicVariable":
        """
        define_public_variable(initial_value, apply_scaling, name, factory) -> PondPublicVariable

        Define a public variable.

        This is like defining a variable in tensorflow except it creates one that can be used by
        the protocol.

        For most cases, you can think of this as the same as the one from tensorflow
        and you don't generally need to consider the difference.

        For those curious, under the hood, the major difference is that this function will pin your
        data to a specific device which will be used to optimize the graph later on.

        :see: tf.Variable

        :param Union[np.ndarray,tf.Tensor,PondPublicTensor] initial_value: The initial value.
        :param bool apply_scaling: Whether or not to scale the value.
        :param str name: What name to give to this node in the graph.
        :param AbstractFactory factory: Which tensor type to represent this value with.
        """
        assert isinstance(
            initial_value, (np.ndarray, tf.Tensor, PondPublicTensor)
        ), type(initial_value)

        factory = factory or self.tensor_factory

        with tf.name_scope("public-var{}".format("-" + name if name else "")):

            if isinstance(initial_value, np.ndarray):
                v = self._encode(initial_value, apply_scaling)
                v_on_0, v_on_1 = v, v

            elif isinstance(initial_value, tf.Tensor):
                v = self._encode(initial_value, apply_scaling, tf_int_type=factory.native_type)
                v_on_0, v_on_1 = v, v

            elif isinstance(initial_value, PondPublicTensor):
                v_on_0, v_on_1 = initial_value.unwrapped

            else:
                raise TypeError(
                    "Don't know how to turn {} into public variable".format(
                        type(initial_value)
                    )
                )

            with tf.device(self.server_0.device_name):
                x_on_0 = factory.variable(v_on_0)

            with tf.device(self.server_1.device_name):
                x_on_1 = factory.variable(v_on_1)

        x = PondPublicVariable(self, x_on_0, x_on_1, apply_scaling)
        _initializers.append(x.initializer)
        return x

    def define_private_variable(
        self,
        initial_value: Union[np.ndarray, tf.Tensor, "PondPublicTensor", "PondPrivateTensor"],
        apply_scaling: bool = True,
        name: Optional[str] = None,
        factory: Optional[AbstractFactory] = None,
    ) -> "PondPrivateVariable":
        """
        define_private_variable(initial_value, apply_scaling, name, factory) -> PondPrivateVariable

        Define a private variable.

        This will take the passed value and construct shares that will be split up between
        those involved in the computationself.

        For example, in a two party architecture, this will split the value into two sets of
        shares and transfer them between each party in a secure manner.

        :see tf.Variable

        :param Union[np.ndarray,tf.Tensor,PondPublicTensor] initial_value: The initial value.
        :param bool apply_scaling: Whether or not to scale the value.
        :param str name: What name to give to this node in the graph.
        :param AbstractFactory factory: Which tensor type to represent this value with.
        """
        assert isinstance(
            initial_value, (np.ndarray, tf.Tensor, PondPublicTensor, PondPrivateTensor)
        ), type(initial_value)

        factory = factory or self.tensor_factory

        with tf.name_scope("private-var{}".format("-" + name if name else "")):

            if isinstance(initial_value, np.ndarray):
                v = factory.tensor(self._encode(initial_value, apply_scaling))
                v0, v1 = self._share(v)

            elif isinstance(initial_value, tf.Tensor):
                v = factory.tensor(self._encode(initial_value, apply_scaling,
                                                tf_int_type=factory.native_type))
                v0, v1 = self._share(v)

            elif isinstance(initial_value, PondPublicTensor):
                v_on_0, _ = initial_value.unwrapped
                with tf.device(self.server_0.device_name):
                    # NOTE[Morten]
                    # we can alternatively avoid transfer of v1 from server0 and server1
                    # by having the crypto producer (pre-)generate sharings of zero
                    v0, v1 = self._share(v_on_0)

            elif isinstance(initial_value, PondPrivateTensor):
                v0, v1 = initial_value.unwrapped

            else:
                raise TypeError(
                    "Don't know how to turn {} into private variable".format(
                        type(initial_value)
                    )
                )

            with tf.device(self.server_0.device_name):
                x0 = factory.variable(v0)

            with tf.device(self.server_1.device_name):
                x1 = factory.variable(v1)

        x = PondPrivateVariable(self, x0, x1, apply_scaling)
        _initializers.append(x.initializer)
        return x

    def define_public_input(
        self,
        player: Union[str, Player],
        inputter_fn: TFEInputter,
        apply_scaling: bool = True,
        name: Optional[str] = None,
    ) -> Union["PondPublicTensor", List["PondPublicTensor"]]:
        """
        define_public_input(player, inputter_fn, apply_scaling, name) -> PondPublicTensor(s)

        Define a public input.

        This represents a `public` input owned by the specified player into the graph.

        :param Union[str,Player] player: Which player owns this input.
        :param bool apply_scaling: Whether or not to scale the value.
        :param str name: What name to give to this node in the graph.
        """

        if isinstance(player, str):
            player = get_config().get_player(player)
        assert isinstance(player, Player)

        factory = self.tensor_factory

        def helper(v: tf.Tensor) -> "PondPublicTensor":
            assert (
                v.shape.is_fully_defined()
            ), "Shape of input '{}' on '{}' is not fully defined".format(
                name if name else "", player.name
            )
            w = factory.tensor(self._encode(v, apply_scaling,
                                            tf_int_type=factory.native_type))
            return PondPublicTensor(self, w, w, apply_scaling)

        with tf.name_scope("public-input{}".format("-" + name if name else "")):

            with tf.device(player.device_name):

                inputs = inputter_fn()

                if isinstance(inputs, tf.Tensor):
                    # single input -> single output
                    v = inputs
                    return helper(v)

                elif isinstance(inputs, (list, tuple)):
                    # multiple inputs -> multiple outputs
                    return [helper(v) for v in inputs]

                else:
                    raise TypeError(
                        "Don't know how to handle inputs of type {}".format(
                            type(inputs)
                        )
                    )

    def define_private_input(
        self,
        player: Union[str, Player],
        inputter_fn: TFEInputter,
        apply_scaling: bool = True,
        name: Optional[str] = None,
        masked: bool = False,
        factory: Optional[AbstractFactory] = None,
    ) -> Union[
        "PondPrivateTensor",
        "PondMaskedTensor",
        List[Union["PondPrivateTensor", "PondMaskedTensor"]],
    ]:
        """
        define_private_input(player, inputter_fn, apply_scaling, name, masked, factory) -> PondPrivateTensor(s)

        Define a private input.

        This represents a `private` input owned by the specified player into the graph.

        :param Union[str,Player] player: Which player owns this input.
        :param bool apply_scaling: Whether or not to scale the value.
        :param str name: What name to give to this node in the graph.
        :param bool masked: Whether or not to mask the input.
        :param AbstractFactory factory: Which backing type to use for this input (e.g. `int100` or `int64`).
        """  # noqa:E501

        factory = factory or self.tensor_factory

        if isinstance(player, str):
            player = get_config().get_player(player)
        assert isinstance(player, Player)

        def helper(v: tf.Tensor) -> Union["PondPrivateTensor", "PondMaskedTensor"]:
            assert (
                v.shape.is_fully_defined()
            ), "Shape of input '{}' on '{}' is not fully defined".format(
                name if name else "", player.name
            )

            w = factory.tensor(self._encode(v, apply_scaling,
                                            tf_int_type=factory.native_type))
            x = self._share_and_wrap(w, apply_scaling)

            if not masked:
                return x
            else:
                with tf.name_scope("local_mask"):
                    a0 = factory.sample_uniform(v.shape)
                    a1 = factory.sample_uniform(v.shape)
                    a = a0 + a1
                    alpha = w - a
                return PondMaskedTensor(self, x, a, a0, a1, alpha, alpha, apply_scaling)

        with tf.name_scope("private-input{}".format("-" + name if name else "")):

            with tf.device(player.device_name):

                inputs = inputter_fn()

                if isinstance(inputs, tf.Tensor):
                    # single input -> single output
                    v = inputs
                    output = helper(v)

                elif isinstance(inputs, (list, tuple)):
                    # multiple inputs -> multiple outputs
                    output = [helper(v) for v in inputs]

                else:
                    raise TypeError(
                        "Don't know how to handle inputs of type {}".format(
                            type(inputs)
                        )
                    )

        return output

    def define_output(
        self,
        player: Union[str, Player],
        xs: Union["PondPrivateTensor", List["PondPrivateTensor"]],
        outputter_fn: Callable[..., Any],
        name: Optional[str] = None,
    ) -> tf.Operation:
        """
        define_output(player, xs, outputter_fn, name) -> tensorflow.Operation

        Define an output for this graph.

        :param Union[str,Player] player: Which player/device this output will be sent to.
        """

        if isinstance(player, str):
            player = get_config().get_player(player)
        assert isinstance(player, Player)

        def helper(x: Union["PondPrivateTensor", "PondMaskedTensor"]) -> tf.Tensor:
            if isinstance(x, PondMaskedTensor):
                x = x.unmasked
            assert isinstance(
                x, PondPrivateTensor
            ), "Don't know how to handle inputs of type {}".format(type(x))
            x0, x1 = x.unwrapped
            w = self._reconstruct(x0, x1)
            v = self._decode(w, x.is_scaled)
            return v

        with tf.name_scope("output{}".format("-" + name if name else "")):

            with tf.device(player.device_name):

                if isinstance(xs, (list, tuple)):
                    op = outputter_fn(*[helper(x) for x in xs])

                else:
                    op = outputter_fn(helper(xs))

                # wrap in tf.group to prevent sending back any tensors (which might hence be leaked)
                op = tf.group(op)

        return op

    @property
    def initializer(self) -> tf.Operation:
        return tf.group(*_initializers)

    def clear_initializers(self) -> None:
        del _initializers[:]

    def _encode(self,
                rationals: Union[tf.Tensor, np.ndarray],
                apply_scaling: bool,
                tf_int_type=None,
                ) -> Union[tf.Tensor, np.ndarray]:
        """
        Encode tensor of rational numbers into tensor of ring elements. Output is of same type
        as input to allow function to be used for constants.
        """

        with tf.name_scope("encode"):

            # we first scale as needed

            if apply_scaling:
                scaled = rationals * self.fixedpoint_config.scaling_factor
            else:
                scaled = rationals

            # and then we round to integers

            if isinstance(scaled, np.ndarray):
                integers = scaled.astype(int).astype(object)

            elif isinstance(scaled, tf.Tensor):
                tf_int_type = tf_int_type or (scaled.dtype
                                              if scaled.dtype in TF_INT_TYPES
                                              else self.tensor_factory.native_type)
                assert tf_int_type in TF_INT_TYPES
                integers = tf.cast(scaled, dtype=tf_int_type)

            else:
                raise TypeError("Don't know how to encode {}".format(type(rationals)))

            assert type(rationals) == type(integers)
            return integers

    @memoize
    def _decode(
        self, elements: AbstractTensor, is_scaled: bool
    ) -> Union[tf.Tensor, np.ndarray]:
        """ Decode tensor of ring elements into tensor of rational numbers """

        with tf.name_scope("decode"):

            bound = self.fixedpoint_config.bound_single_precision
            scaled = (elements + bound).to_native() - bound

            if is_scaled:
                return scaled / self.fixedpoint_config.scaling_factor
            else:
                return scaled

    def _share(self, secret: AbstractTensor) -> Tuple[AbstractTensor, AbstractTensor]:

        with tf.name_scope("share"):
            share0 = secret.factory.sample_uniform(secret.shape)
            share1 = secret - share0

            # randomized swap to distribute load between two servers wrt who gets the seed
            if random() < 0.5:
                share0, share1 = share1, share0

        return share0, share1

    def _share_and_wrap(self, secret: AbstractTensor, is_scaled: bool) -> "PondPrivateTensor":
        s0, s1 = self._share(secret)
        return PondPrivateTensor(self, s0, s1, is_scaled)

    def _reconstruct(
        self, share0: AbstractTensor, share1: AbstractTensor
    ) -> AbstractTensor:
        with tf.name_scope("reconstruct"):
            return share0 + share1

    @memoize
    def assign(self, variable: "PondPrivateVariable", value) -> tf.Operation:
        assert isinstance(variable, PondPrivateVariable), type(variable)
        assert isinstance(value, PondPrivateTensor), type(value)
        assert (
            variable.is_scaled == value.is_scaled
        ), "Scaling must match: {}, {}".format(variable.is_scaled, value.is_scaled)

        var0, var1 = variable.variable0, variable.variable1
        val0, val1 = value.share0, value.share1

        with tf.name_scope("assign"):

            with tf.device(self.server_0.device_name):
                op0 = var0.assign_from_same(val0)

            with tf.device(self.server_1.device_name):
                op1 = var1.assign_from_same(val1)

            op = tf.group(op0, op1)

        return op

    @memoize
    def add(self, x, y):
        """
        add(x, y) -> PondTensor

        Adds two tensors `x` and `y`.

        :param PondTensor x: The first operand.
        :param PondTensor y: The second operand.
        """
        x, y = self.lift(x, y)
        return self.dispatch("add", x, y)

    def lift(
        self, x, y=None, apply_scaling: Optional[bool] = None
    ) -> Union["PondTensor", Tuple["PondTensor", "PondTensor"]]:
        """
        lift(x, y=None, apply_scaling=None) -> PondTensor(s)

        Convenience method for working with mixed typed tensors in programs:
        combining any of the Pond objects together with e.g. ints and floats
        will automatically lift the latter into Pond objects.

        :param int,float,PondTensor x: Python object to lift.
        :param int,float,PondTensor y: Second Python object to lift, optional.
        :param bool apply_scaling: Whether to apply scaling to the input object(s).
        """

        if y is None:

            if isinstance(x, (int, float)):
                return self.define_constant(np.array([x]))

            if isinstance(x, PondTensor):
                return x

            raise TypeError("Don't know how to lift {}".format(type(x)))

        else:

            if isinstance(x, (int, float)):

                if isinstance(y, (int, float)):
                    x = self.define_constant(np.array([x]))
                    y = self.define_constant(np.array([y]))
                    return x, y

                if isinstance(y, PondTensor):
                    x = self.define_constant(
                        np.array([x]),
                        apply_scaling=apply_scaling or y.is_scaled,
                        factory=y.backing_dtype,
                    )
                    return x, y

                raise TypeError(
                    "Don't know how to lift {}, {}".format(type(x), type(y))
                )

            if isinstance(x, PondTensor):
                if isinstance(y, (int, float)):
                    y = self.define_constant(
                        np.array([y]),
                        apply_scaling=apply_scaling or x.is_scaled,
                        factory=x.backing_dtype,
                    )
                    return x, y

                if isinstance(y, PondTensor):
                    return x, y

                raise TypeError(
                    "Don't know how to lift {}, {}".format(type(x), type(y))
                )

            raise TypeError("Don't know how to lift {}, {}".format(type(x), type(y)))

    @memoize
    def add_n(self, tensors):
        # TODO(Morten) we could optimize by doing lazy reductions, potentially segmenting as needed
        return reduce(lambda x, y: x + y, tensors)

    @memoize
    def reduce_sum(self, x, axis=None, keepdims=None):
        x = self.lift(x)
        return self.dispatch("reduce_sum", x, axis=axis, keepdims=keepdims)

    def sum(self, x, axis=None, keepdims=None):
        return self.reduce_sum(x, axis, keepdims)

    @memoize
    def cumsum(self, x, axis=0, exclusive=False, reverse=False):
        return self.dispatch(
            "cumsum", x, axis=axis, exclusive=exclusive, reverse=reverse
        )

    @memoize
    def sub(self, x, y):
        x, y = self.lift(x, y)
        return self.dispatch("sub", x, y)

    def mask(self, x):
        if isinstance(x, (list, tuple)):
            # apply recursively
            return [self.mask(xi) for xi in x]

        node_key = ("mask", x)
        x_masked = nodes.get(node_key, None)

        if x_masked is not None:
            return x_masked

        if isinstance(x, PondPrivateTensor):
            x_masked = _mask_private(self, x)

        else:
            raise TypeError("Don't know how to mask {}".format(type(x)))

        nodes[node_key] = x_masked
        return x_masked

    @memoize
    def mul(self, x, y):
        x, y = self.lift(x, y)
        return self.dispatch("mul", x, y)

    @memoize
    def square(self, x):
        return self.dispatch("square", x)

    @memoize
    def matmul(self, x: "PondTensor", y: "PondTensor") -> "PondTensor":
        return self.dispatch("matmul", x, y)

    def dot(self, x, y):
        return self.matmul(x, y)

    @memoize
    def div(self, x, y):
        """
        Performs a true division of `x` by `y` where `y` is public.

        No flooring is performing if `y` is an integer type as it is implicitly treated as a float.
        """

        assert isinstance(x, PondTensor)

        if isinstance(y, float):
            y_inverse = 1. / y
        if isinstance(y, int):
            y_inverse = 1. / float(y)
        elif isinstance(y, PondPublicTensor):
            y_inverse = 1. / y.decode()
        else:
            raise TypeError("Don't know how to divide by type {}".format(type(y)))

        return self.mul(x, y_inverse)

    @memoize
    def truncate(self, x: "PondTensor"):
        return self.dispatch("truncate", x)

    @memoize
    def indexer(self, x: "PondTensor", slice: Union[Slice, Ellipse]) -> "PondTensor":
        return self.dispatch("indexer", x, slice)

    def transpose(self, x, perm=None) -> "PondTensor":
        """
        transpose(x, perm=None) -> PondTensor

        Transposes the input `x`, or permutes the axes of `x` if `perm` is given.

        :param PondTensor x: The tensor to transpose or permute.
        :param List perm: A permutation of axis indices.
        """

        node_key = ("transpose", x)
        x_t = nodes.get(node_key, None)

        if x_t is not None:
            return x_t

        if isinstance(x, PondPublicTensor):
            x_t = _transpose_public(self, x, perm=perm)

        elif isinstance(x, PondPrivateTensor):
            x_t = _transpose_private(self, x, perm=perm)

        elif isinstance(x, PondMaskedTensor):
            x_t = _transpose_masked(self, x, perm=perm)

        else:
            raise TypeError("Don't know how to transpose {}".format(type(x)))

        nodes[node_key] = x_t
        return x_t

    @memoize
    def reshape(self, x: "PondTensor", shape: List[int]):
        """
        reshape(x, shape) -> PondTensor

        Reshape `x` into a tensor with a new `shape`.

        :param PondTensor x: Input tensor.
        :param (int,...) shape: Shape of output tensor.
        """

        if isinstance(x, PondPublicTensor):
            return _reshape_public(self, x, shape)

        if isinstance(x, PondPrivateTensor):
            return _reshape_private(self, x, shape)

        if isinstance(x, PondMaskedTensor):
            return _reshape_masked(self, x, shape)

        raise TypeError("Don't know how to reshape {}".format(type(x)))

    @memoize
    def negative(self, x: "PondTensor"):
        """
        negative(x) -> PondTensor

        Computes numerical negative value element-wise.

        :param PondTensor x: Input tensor.
        """
        return self.dispatch("negative", x)

    @memoize
    def expand_dims(self, x: "PondTensor", axis=None):

        if isinstance(x, PondPublicTensor):
            return _expand_dims_public(self, x, axis=axis)

        if isinstance(x, PondPrivateTensor):
            return _expand_dims_private(self, x, axis=axis)

        if isinstance(x, PondMaskedTensor):
            return _expand_dims_masked(self, x, axis=axis)

        raise TypeError("Don't know how to expand dims {}".format(type(x)))

    @memoize
    def squeeze(self, x: "PondTensor", axis: Optional[List[int]] = None):

        if isinstance(x, PondPublicTensor):
            return _squeeze_public(self, x, axis)

        if isinstance(x, PondPrivateTensor):
            return _squeeze_private(self, x, axis)

        if isinstance(x, PondMaskedTensor):
            return _squeeze_masked(self, x, axis)

        raise TypeError("Don't know how to squeeze {}".format(type(x)))

    def strided_slice(self, x: "PondTensor", *args: Any, **kwargs: Any):
        """
        strided_slice(x, *args, **kwargs) -> PondTensor

        See https://www.tensorflow.org/api_docs/python/tf/strided_slice for further documentation.
        """

        node_key = ("strided_slice", x)

        x_sliced = nodes.get(node_key, None)

        if x_sliced is not None:
            return x_sliced

        if isinstance(x, PondPublicTensor):
            x_sliced = _strided_slice_public(self, x, args, kwargs)
        elif isinstance(x, PondPrivateTensor):
            x_sliced = _strided_slice_private(self, x, args, kwargs)
        elif isinstance(x, PondMaskedTensor):
            x_sliced = _strided_slice_masked(self, x, args, kwargs)
            nodes[("strided_slice", x.unmasked)] = x_sliced.unmasked
        else:
            raise TypeError("Don't know how to do a strided slice {}".format(type(x)))

        nodes[node_key] = x_sliced

        return x_sliced

    @memoize
    def split(
        self, x: "PondTensor", num_split: int, axis: int = 0
    ) -> List["PondTensor"]:
        return self.dispatch("split", x, num_split, axis=axis)

    def stack(self, xs: List["PondTensor"], axis: int = 0):

        node_key = ("stack", tuple(xs))
        xs_stack = nodes.get(node_key, None)

        if xs_stack is not None:
            return xs_stack

        if all([isinstance(x, PondPublicTensor) for x in xs]):
            xs_stack = _stack_public(self, xs, axis=axis)

        elif all([isinstance(x, PondPrivateTensor) for x in xs]):
            xs_stack = _stack_private(self, xs, axis=axis)

        elif all([isinstance(x, PondMaskedTensor) for x in xs]):
            xs_stack = _stack_masked(self, xs, axis=axis)
        else:
            raise TypeError("Don't know how to do a stack {}".format(type(xs)))

        nodes[node_key] = xs_stack

        return xs_stack

    @memoize
    def concat(self, xs: List["PondTensor"], axis):

        if all(isinstance(x, PondPublicTensor) for x in xs):
            return _concat_public(self, xs, axis=axis)

        if all(isinstance(x, PondPrivateTensor) for x in xs):
            return _concat_private(self, xs, axis=axis)

        if all(isinstance(x, PondMaskedTensor) for x in xs):
            return _concat_masked(self, xs, axis=axis)

        raise TypeError("Don't know how to do a concat {}".format(type(xs)))

    @memoize
    def sigmoid(self, x: "PondTensor"):
        assert isinstance(x, PondTensor), type(x)

        w0 = 0.5
        w1 = 0.2159198015
        w3 = -0.0082176259
        w5 = 0.0001825597
        w7 = -0.0000018848
        w9 = 0.0000000072

        with tf.name_scope("sigmoid"):

            # TODO[Morten] try in single round
            x1 = x
            x2 = x1.square()
            x3 = x2 * x
            x5 = x2 * x3
            x7 = x2 * x5
            x9 = x2 * x7

            y1 = x1 * w1
            y3 = x3 * w3
            y5 = x5 * w5
            y7 = x7 * w7
            y9 = x9 * w9

            z = y9 + y7 + y5 + y3 + y1 + w0
            # z = y7 + y5 + y3 + y1 + w0

        return z

    @memoize
    def relu(self, x: "PondTensor"):
        assert isinstance(x, PondTensor), type(x)

        w0 = 0.44015372000819103
        w1 = 0.500000000
        w2 = 0.11217537671414643
        w4 = -0.0013660836712429923
        w6 = 9.009136367360004e-06
        w8 = -2.1097433984e-08

        with tf.name_scope("relu"):

            x1 = x
            x2 = x.square()
            x4 = x2 * x2
            x6 = x2 * x4
            x8 = x2 * x6

            y1 = x1 * w1
            y2 = x2 * w2
            y4 = x4 * w4
            y6 = x6 * w6
            y8 = x8 * w8

            z = y8 + y6 + y4 + y2 + y1 + w0

        return z

    @memoize
    def tanh(self, x: "PondTensor"):
        assert isinstance(x, PondTensor), type(x)

        w0 = 0.0
        w1 = 0.852721056
        w3 = -0.12494112
        w5 = 0.010654528
        w7 = -0.000423424

        with tf.name_scope("relu"):

            x1 = x
            x2 = x.square()
            x3 = x2 * x1
            x5 = x2 * x3
            x7 = x2 * x5

            y1 = x1 * w1
            y3 = x3 * w3
            y5 = x5 * w5
            y7 = x7 * w7

            z = y7 + y5 + y3 + y1 + w0

        return z

    @memoize
    def reveal(self, x):
        return self.dispatch("reveal", x)

    def cache(self, x):

        if isinstance(x, (list, tuple)):
            # apply recursively
            return [self.cache(xi) for xi in x]

        node_key = ("cache", x)
        cached = nodes.get(node_key, None)

        if cached is not None:
            return cached

        dispatch = {
            PondPublicTensor: _cache_public,
            PondPrivateTensor: _cache_private,
            PondMaskedTensor: _cache_masked,
        }
        func = dispatch.get(_type(x), None)
        if func is None:
            raise TypeError("Don't know how to cache {}".format(type(x)))

        cached = func(self, x)
        nodes[node_key] = cached

        return cached

    def conv2d(self, x, w, strides, padding):

        node_key = ("conv2d", x, w, strides, padding)
        z = nodes.get(node_key, None)

        if z is not None:
            return z

        dispatch = {
            (PondPublicTensor, PondPublicTensor): _conv2d_public_public,
            (PondPublicTensor, PondPrivateTensor): _conv2d_public_private,
            (PondPublicTensor, PondMaskedTensor): _conv2d_public_masked,
            (PondPrivateTensor, PondPublicTensor): _conv2d_private_public,
            (PondPrivateTensor, PondPrivateTensor): _conv2d_private_private,
            (PondPrivateTensor, PondMaskedTensor): _conv2d_private_masked,
            (PondMaskedTensor, PondPublicTensor): _conv2d_masked_public,
            (PondMaskedTensor, PondPrivateTensor): _conv2d_masked_private,
            (PondMaskedTensor, PondMaskedTensor): _conv2d_masked_masked,
        }

        func = dispatch.get((_type(x), _type(w)), None)
        if func is None:
            raise TypeError(
                "Don't know how to conv2d {} and {}".format(type(x), type(w))
            )

        z = func(self, x, w, strides, padding)
        nodes[node_key] = z

        return z

    def maxpool2d(self, x, pool_size, strides, padding):
        raise NotImplementedError("Only SecureNN supports Max Pooling")

    def avgpool2d(self, x, pool_size, strides, padding):
        node_key = ("avgpool2d", x, tuple(pool_size), tuple(strides), padding)
        z = nodes.get(node_key, None)

        if z is not None:
            return z

        dispatch = {
            PondPublicTensor: _avgpool2d_public,
            PondPrivateTensor: _avgpool2d_private,
            PondMaskedTensor: _avgpool2d_masked,
        }

        func = dispatch.get(_type(x), None)
        if func is None:
            raise TypeError("Don't know how to avgpool2d {}".format(type(x)))

        z = func(self, x, pool_size, strides, padding)
        nodes[node_key] = z

        return z

    def batch_to_space_nd(self, x, block_shape, crops):
        return self.dispatch("batch_to_space_nd", x, block_shape, crops)

    def space_to_batch_nd(self, x, block_shape, paddings):
        return self.dispatch("space_to_batch_nd", x, block_shape, paddings)

    @memoize
    def equal(self, x, y):
        x, y = self.lift(x, y)
        return self.dispatch("equal", x, y)

    def dispatch(self, base_name, *args, container=None, **kwargs):
        func_name = "_{}_{}".format(
            base_name,
            "_".join([arg.dispatch_id for arg in args if hasattr(arg, "dispatch_id")]),
        )

        if container is None:
            container = _thismodule

        func = getattr(container, func_name, None)
        if func is not None:
            return func(self, *args, **kwargs)
        else:
            raise TypeError(
                "Don't know how to {}: {}".format(
                    base_name, [type(arg) for arg in args]
                )
            )

    def pad(self, x: 'PondTensor', paddings: list):

        backing_type = x.backing_dtype

        if isinstance(x, PondPublicTensor):
            tensor_type = PondPublicTensor
        elif isinstance(x, PondPrivateTensor):
            tensor_type = PondPrivateTensor
        else:
            raise ValueError("Don't know how to handle type {}".format(type(x)))

        def zeros(shape):
            # NOTE
            # this is a cheating way of getting zeros for the case where tensor_type is private,
            # in particular because non-interactive truncation may fail if applied to these
            # tensors only; for this reason we here use the assumption that truncation will only
            # ever be applied after these zeros have been mix with proper shares

            with tf.device(self.server_0.device_name):
                zeros0 = backing_type.tensor(
                    tf.zeros(shape, dtype=backing_type.native_type)
                )

            with tf.device(self.server_1.device_name):
                zeros1 = backing_type.tensor(
                    tf.zeros(shape, dtype=backing_type.native_type)
                )

            return tensor_type(self, zeros0, zeros1, True)

        def prepend_zeros(tensor, pad_amt, axis):

            with tf.name_scope('prepend'):

                if pad_amt == 0:
                    return tensor

                padshape = tuple(
                    dim if i != axis else pad_amt
                    for (i, dim) in enumerate(tensor.shape.as_list())
                )

                return self.concat([zeros(padshape), tensor], axis=axis)

        def append_zeros(tensor, pad_amt, axis):

            with tf.name_scope('append'):

                if pad_amt == 0:
                    return tensor

                padshape = tuple(
                    dim if i != axis else pad_amt
                    for (i, dim) in enumerate(tensor.shape.as_list())
                )

                return self.concat([tensor, zeros(padshape)], axis=axis)

        with tf.name_scope('pad'):
            for axis, (pad_before, pad_after) in enumerate(paddings):
                x = append_zeros(x, pad_after, axis)
                x = prepend_zeros(x, pad_before, axis)

        return x


#
# Classes representing the base values in the Pond protocol.
#


class PondTensor(abc.ABC):
    """
    This class functions mostly as a convenient way of exposing operations
    directly on the various tensor objects, ie allowing one to write `x + y`
    instead of `prot.add(x, y)`. Since this functionality is shared among all
    tensors we put it in this superclass.

    This class should never be instantiated on its own.
    Instead you should use your chosen protocols factory methods::

        x = prot.define_private_input(tf.constant(np.array([1,2,3,4])))
        y = prot.define_public_input(tf.constant(np.array([4,5,6,7])))

        z = x + y

        with config.Session() as sess:
            answer = z.reveal().eval(sess)

            print(answer) # => [5, 7, 9, 11]
    """

    def __init__(self, prot, is_scaled):
        self.prot = prot
        self.is_scaled = is_scaled

    @property
    @abc.abstractmethod
    def shape(self) -> List[int]:
        """
        :rtype: List[int]
        :returns: The shape of this tensor.
        """
        pass

    @property
    @abc.abstractmethod
    def unwrapped(self) -> Tuple[AbstractTensor, ...]:
        pass

    def add(self, other):
        """
        Add `other` to this PondTensor.  This can be another tensor with the same
        backing or a primitive.

        This function returns a new PondTensor and does not modify this one.

        :param PondTensor other: a or primitive (e.g. a float)
        :return: A new PondTensor with `other` added.
        :rtype: PondTensor
        """
        return self.prot.add(self, other)

    def __add__(self, other):
        """
        See :meth:`~tensorflow_encrypted.protocol.pond.PondTensor.add`
        """
        return self.prot.add(self, other)

    def __radd__(self, other):
        return other.prot.add(self, other)

    def reduce_sum(self, axis=None, keepdims=None):
        """
        Like :meth:`tensorflow.reduce_sum`

        :param int axis:  The axis to reduce along
        :param bool keepdims: If true, retains reduced dimensions with length 1.
        :return: A new PondTensor
        :rtype: PondTensor
        """
        return self.prot.reduce_sum(self, axis, keepdims)

    def sum(self, axis=None, keepdims=None):
        """
        See :meth:`PondTensor.reduce_sum`
        """
        return self.reduce_sum(axis, keepdims)

    def sub(self, other):
        """
        Subtract `other` from this tensor.

        :param PondTensor other: to subtract
        :return: A new PondTensor
        :rtype: PondTensor
        """
        return self.prot.sub(self, other)

    def __sub__(self, other):
        return self.prot.sub(self, other)

    def __rsub__(self, other):
        return self.prot.sub(self, other)

    def mul(self, other):
        """
        Multiply this tensor with `other`

        :param PondTensor other: to multiply
        :return: A new PondTensor
        :rtype: PondTensor
        """
        return self.prot.mul(self, other)

    def __mul__(self, other):
        return self.prot.mul(self, other)

    def __rmul__(self, other):
        return self.prot.mul(self, other)

    def __truediv__(self, other):
        return self.prot.div(self, other)

    def __mod__(self, other):
        return self.prot.mod(self, other)

    def square(self):
        """
        Square this tensor.

        :return: A new PondTensor
        :rtype: PondTensor
        """
        return self.prot.square(self)

    def matmul(self, other):
        """
        MatMul this tensor with `other`.  This will perform matrix multiplication,
        rather than elementwise like :meth:`~tensorflow_encrypted.protocol.pond.PondTensor.mul`

        :param PondTensor other: to subtract
        :return: A new PondTensor
        :rtype: PondTensor
        """
        return self.prot.matmul(self, other)

    def dot(self, other):
        """
        Alias for :meth:`~tensorflow_encrypted.protocol.pond.PondTensor.matmul`

        :return: A new PondTensor
        :rtype: PondTensor
        """
        return self.matmul(other)

    def __getitem__(self, slice):
        return self.prot.indexer(self, slice)

    def transpose(self, perm=None):
        """
        Transpose this tensor.

        See :meth:`tensorflow.transpose`

        :param List[int]: A permutation of the dimensions of this tensor.

        :return: A new PondTensor
        :rtype: PondTensor
        """
        return self.prot.transpose(self, perm)

    def truncate(self):
        """
        Truncate this tensor.

        `TODO`

        :return: A new PondTensor
        :rtype: PondTensor
        """
        return self.prot.truncate(self)

    def expand_dims(self, axis=None):
        """
        :See: tf.expand_dims

        :return: A new PondTensor
        :rtype: PondTensor
        """
        return self.prot.expand_dims(self, axis=axis)

    def reshape(self, shape: List[int]) -> "PondTensor":
        """
        :See: tf.reshape

        :param List[int] shape: The new shape of the tensor.
        :rtype: PondTensor
        :returns: A new tensor with the contents of this tensor, but with the new specified shape.
        """
        return self.prot.reshape(self, shape)

    def negative(self) -> "PondTensor":
        """
        :See: tf.negative

        :rtype: PondTensor
        :returns: A new tensor with numerical negative value element-wise computed.
        """
        return self.prot.negative(self)

    def reduce_max(self, axis: int) -> "PondTensor":
        """
        :See: tf.reduce_max

        :param int axis: The axis to take the max along
        :rtype: PondTensor
        :returns: A new pond tensor with the max value from each axis.
        """
        return self.prot.reduce_max(self, axis)


class PondPublicTensor(PondTensor):
    """
    This class represents a public tensor, known by at least the two servers
    but potentially known by more. Although there is only a single value we
    replicate it on both servers to avoid sending it from one to the other
    in the operations where it's needed by both (eg multiplication).
    """

    dispatch_id = "public"

    def __init__(
        self,
        prot: Pond,
        value_on_0: AbstractTensor,
        value_on_1: AbstractTensor,
        is_scaled: bool,
    ) -> None:
        assert isinstance(value_on_0, AbstractTensor), type(value_on_0)
        assert isinstance(value_on_1, AbstractTensor), type(value_on_1)
        assert value_on_0.shape == value_on_1.shape

        super(PondPublicTensor, self).__init__(prot, is_scaled)
        self.value_on_0 = value_on_0
        self.value_on_1 = value_on_1

    def __repr__(self) -> str:
        return "PondPublicTensor(shape={})".format(self.shape)

    @property
    def shape(self) -> List[int]:
        return self.value_on_0.shape

    @property
    def backing_dtype(self):
        return self.value_on_0.factory

    @property
    def unwrapped(self) -> Tuple[AbstractTensor, ...]:
        """
        Unwrap the tensor.

        This will return the value for each of the parties that collectively own the tensor.

        In most cases, this will be the same value on each device.

        .. code-block:: python

            x_0, y_0 = tensor.unwrapped
            # x_0 == 10 with the value pinned to player_0's device.
            # y_0 == 10 with the value pinned to player_1's device.

        In most cases you will want to work on this data on the specified device.

        .. code-block:: python

            x_0, y_0 = tensor.unwrapped

            with tf.device(prot.player_0.device_name):
                # act on x_0

            with tf.device(prot.player_1.device_name):
                # act on y_0

        In most cases you will not need to use this method.  All funtions
        will hide this functionality for you (e.g. `add`, `mul`, etc).
        """
        return (self.value_on_0, self.value_on_1)

    def decode(self) -> Union[np.ndarray, tf.Tensor]:
        return self.prot._decode(self.value_on_0, self.is_scaled)


class PondPrivateTensor(PondTensor):
    """
    This class represents a private value that may be unknown to everyone.
    """

    dispatch_id = "private"

    def __init__(
        self,
        prot: Pond,
        share0: AbstractTensor,
        share1: AbstractTensor,
        is_scaled: bool,
    ) -> None:
        assert isinstance(share0, AbstractTensor), type(share0)
        assert isinstance(share1, AbstractTensor), type(share1)
        assert share0.shape == share1.shape

        super(PondPrivateTensor, self).__init__(prot, is_scaled)
        self.share0 = share0
        self.share1 = share1

    def __repr__(self) -> str:
        return "PondPrivateTensor(shape={})".format(self.shape)

    @property
    def shape(self) -> List[int]:
        return self.share0.shape

    @property
    def backing_dtype(self):
        return self.share0.factory

    @property
    def unwrapped(self) -> Tuple[AbstractTensor, ...]:
        """
        Unwrap the tensor.

        This will return the shares for each of the parties that collectively own the tensor.

        .. code-block:: python

            x_0, y_0 = tensor.unwrapped
            # x_0 == private shares of the value pinned to player_0's device.
            # y_0 == private shares of the value pinned to player_1's device.

        In most cases you will not need to use this method.  All funtions
        will hide this functionality for you (e.g. `add`, `mul`, etc).
        """
        return (self.share0, self.share1)

    def reveal(self) -> PondPublicTensor:
        return self.prot.reveal(self)


class PondMaskedTensor(PondTensor):
    """
    This class is part of an optimization where values are only ever masked
    once as opposed to for every operation in which they are used. As such
    it represents a private value with additional data associated, namely
    the masks used for the shares on the two servers as well as on the
    crypto provider. For convenience it keeps a reference to the unmasked
    value as well (in the form of a private tensor).
    """

    dispatch_id = "masked"

    def __init__(
        self,
        prot: Pond,
        unmasked: PondPrivateTensor,
        a: AbstractTensor,
        a0: AbstractTensor,
        a1: AbstractTensor,
        alpha_on_0: AbstractTensor,
        alpha_on_1: AbstractTensor,
        is_scaled: bool,
    ) -> None:
        assert isinstance(unmasked, PondPrivateTensor)

        super(PondMaskedTensor, self).__init__(prot, is_scaled)
        self.unmasked = unmasked
        self.a = a
        self.a0 = a0
        self.a1 = a1
        self.alpha_on_0 = alpha_on_0
        self.alpha_on_1 = alpha_on_1

    def __repr__(self) -> str:
        return "PondMaskedTensor(shape={})".format(self.shape)

    @property
    def shape(self) -> List[int]:
        return self.a.shape

    @property
    def backing_dtype(self):
        return self.a.factory

    @property
    def unwrapped(self) -> Tuple[AbstractTensor, ...]:
        return (self.a, self.a0, self.a1, self.alpha_on_0, self.alpha_on_1)

    def reveal(self) -> PondPublicTensor:
        return self.prot.reveal(self.unmasked)


#
# Extentions of the base Pond classes that record extra information
# relevant to how TensorFlow works.
#


class PondConstant(PondPublicTensor):
    """
    This class essentially represents a public value, however it additionally
    records the fact that the underlying value was declared as a constant.
    """

    def __init__(self, prot, constant_on_0, constant_on_1, is_scaled):
        assert isinstance(constant_on_0, AbstractConstant), type(constant_on_0)
        assert isinstance(constant_on_1, AbstractConstant), type(constant_on_1)
        assert constant_on_0.shape == constant_on_1.shape

        super(PondConstant, self).__init__(
            prot, constant_on_0, constant_on_1, is_scaled
        )
        self.constant_on_0 = constant_on_0
        self.constant_on_1 = constant_on_1

    def __repr__(self) -> str:
        return "PondConstant(shape={})".format(self.shape)


class PondPublicPlaceholder(PondPublicTensor):
    """
    This class essentially represents a public value, however it additionally
    records the fact that the backing tensor was declared as a placeholder in
    order to allow treating it as a placeholder itself.
    """

    def __init__(self, prot, placeholder_on_0, placeholder_on_1, is_scaled):
        assert isinstance(placeholder_on_0, AbstractPlaceholder), type(placeholder_on_0)
        assert isinstance(placeholder_on_0, AbstractPlaceholder), type(placeholder_on_1)
        assert placeholder_on_0.shape == placeholder_on_1.shape

        super(PondPublicPlaceholder, self).__init__(
            prot, placeholder_on_0, placeholder_on_1, is_scaled
        )
        self.placeholder_on_0 = placeholder_on_0
        self.placeholder_on_1 = placeholder_on_1

    def __repr__(self) -> str:
        return "PondPublicPlaceholder(shape={})".format(self.shape)


class PondPrivatePlaceholder(PondPrivateTensor):
    """
    This class essentially represents a private value, however it additionally
    records the fact that the backing tensor was declared as a placeholder in
    order to allow treating it as a placeholder itself.
    """

    def __init__(self, prot, placeholder, tensor0, tensor1, is_scaled):
        assert isinstance(placeholder, AbstractPlaceholder), type(placeholder)
        assert isinstance(tensor0, AbstractTensor), type(tensor0)
        assert isinstance(tensor1, AbstractTensor), type(tensor1)
        assert tensor0.shape == tensor1.shape

        super(PondPrivatePlaceholder, self).__init__(prot, tensor0, tensor1, is_scaled)
        self.placeholders = placeholder.backing
        self.tensor0 = tensor0
        self.tensor1 = tensor1

    def __repr__(self) -> str:
        return "PondPrivatePlaceholder(shape={})".format(self.shape)

    def feed(self, value):
        assert isinstance(value, np.ndarray), type(value)

        v = self.prot.tensor_factory.tensor(self.prot._encode(value, self.is_scaled))
        return {p: v for p, v in zip(self.placeholders, v.backing)}


class PondPublicVariable(PondPublicTensor):
    """
    This class essentially represents a public value, however it additionally
    records the fact that the backing tensor was declared as a variable in
    order to allow treating it as a variable itself.
    """

    def __init__(self, prot, variable_on_0, variable_on_1, is_scaled):
        assert isinstance(variable_on_0, AbstractVariable), type(variable_on_0)
        assert isinstance(variable_on_1, AbstractVariable), type(variable_on_1)
        assert variable_on_0.shape == variable_on_1.shape

        super(PondPublicVariable, self).__init__(
            prot, variable_on_0, variable_on_1, is_scaled
        )
        self.variable_on_0 = variable_on_0
        self.variable_on_1 = variable_on_1
        self.initializer = tf.group(
            *[var.initializer for var in [variable_on_0, variable_on_1]]
        )

    def __repr__(self) -> str:
        return "PondPublicVariable(shape={})".format(self.shape)


class PondPrivateVariable(PondPrivateTensor):
    """
    This class essentially represents a private value, however it additionally
    records the fact that the backing tensor was declared as a variable in
    order to allow treating it as a variable itself.
    """

    def __init__(self, prot, variable0, variable1, is_scaled):
        assert isinstance(variable0, AbstractVariable), type(variable0)
        assert isinstance(variable1, AbstractVariable), type(variable1)
        assert variable0.shape == variable1.shape

        super(PondPrivateVariable, self).__init__(prot, variable0, variable1, is_scaled)
        self.variable0 = variable0
        self.variable1 = variable1
        self.initializer = tf.group(
            *[var.initializer for var in [variable0, variable1]]
        )

    def __repr__(self) -> str:
        return "PondPrivateVariable(shape={})".format(self.shape)


class PondCachedPublicTensor(PondPublicTensor):
    def __init__(self, prot, x_on_0, x_on_1, is_scaled, updater):
        assert isinstance(x_on_0, AbstractTensor), type(x_on_0)
        assert isinstance(x_on_1, AbstractTensor), type(x_on_1)
        assert isinstance(updater, tf.Operation), type(updater)

        super(PondCachedPublicTensor, self).__init__(prot, x_on_0, x_on_1, is_scaled)
        self.updater = updater

    def __repr__(self) -> str:
        return "PondCachedPublicTensor(shape={})".format(self.shape)


class PondCachedPrivateTensor(PondPrivateTensor):
    def __init__(self, prot, x0, x1, is_scaled, updater):
        assert isinstance(x0, AbstractTensor), type(x0)
        assert isinstance(x1, AbstractTensor), type(x1)
        assert isinstance(updater, tf.Operation), type(updater)

        super(PondCachedPrivateTensor, self).__init__(prot, x0, x1, is_scaled)
        self.updater = updater

    def __repr__(self) -> str:
        return "PondCachedPrivateTensor(shape={})".format(self.shape)


class PondCachedMaskedTensor(PondMaskedTensor):
    def __init__(
        self, prot, unmasked, a, a0, a1, alpha_on_0, alpha_on_1, is_scaled, updater
    ):
        assert isinstance(unmasked, PondPrivateTensor), type(unmasked)
        assert isinstance(a, AbstractTensor), type(a)
        assert isinstance(a0, AbstractTensor), type(a0)
        assert isinstance(a1, AbstractTensor), type(a1)
        assert isinstance(alpha_on_0, AbstractTensor), type(alpha_on_0)
        assert isinstance(alpha_on_1, AbstractTensor), type(alpha_on_1)
        assert isinstance(updater, tf.Operation), type(updater)

        super(PondCachedMaskedTensor, self).__init__(
            prot, unmasked, a, a0, a1, alpha_on_0, alpha_on_1, is_scaled
        )
        self.updater = updater

    def __repr__(self) -> str:
        return "PondCachedMaskedTensor(shape={})".format(self.shape)


#
# helpers
#


def _type(x):

    if isinstance(x, PondPublicTensor):
        return PondPublicTensor

    if isinstance(x, PondPrivateTensor):
        return PondPrivateTensor

    if isinstance(x, PondMaskedTensor):
        return PondMaskedTensor

    return type(x)


# TODO[Morten] this is just a very first step; far from finished
def debug(x: PondTensor, summarize=None, message=""):
    if isinstance(x, PondPublicTensor):
        tf.print(
            x.value_on_0.value,
            [x.value_on_0.value],
            summarize=summarize,
            message=message,
        )

    elif isinstance(x, PondPrivateTensor):
        tf.print(
            x.share0.value,
            [x.reveal().value_on_0.value],
            summarize=summarize,
            message=message,
        )

    else:
        raise TypeError("Don't know how to debug {}".format(type(x)))


#
# cache
#


def _cache_wrap_helper(prot, sources):
    variables = [
        prot.tensor_factory.variable(
            tf.zeros(shape=source.shape, dtype=prot.tensor_factory.native_type)
        )
        for source in sources
    ]
    updater = tf.group(
        *[var.assign_from_same(val) for var, val in zip(variables, sources)]
    )
    return variables, updater


def _cache_public(prot, x):
    assert isinstance(x, PondPublicTensor), type(x)

    x_on_0, x_on_1 = x.unwrapped

    with tf.name_scope("cache"):

        with tf.device(prot.server_0.device_name):
            [x_on_0_cached], updater0 = _cache_wrap_helper(prot, [x_on_0])

        with tf.device(prot.server_1.device_name):
            [x_on_1_cached], updater1 = _cache_wrap_helper(prot, [x_on_1])

        updater = tf.group(updater0, updater1)

    global_cache_updaters.append(updater)
    return PondCachedPublicTensor(
        prot, x_on_0_cached, x_on_1_cached, x.is_scaled, updater
    )


def _cache_private(prot, x):
    assert isinstance(x, PondPrivateTensor), type(x)

    x0, x1 = x.unwrapped

    with tf.name_scope("cache"):

        with tf.device(prot.server_0.device_name):
            [x0_cached], updater0 = _cache_wrap_helper(prot, [x0])

        with tf.device(prot.server_1.device_name):
            [x1_cached], updater1 = _cache_wrap_helper(prot, [x1])

        updater = tf.group(updater0, updater1)

    global_cache_updaters.append(updater)
    return PondCachedPrivateTensor(prot, x0_cached, x1_cached, x.is_scaled, updater)


def _cache_masked(prot, x):
    assert isinstance(x, PondMaskedTensor), type(x)

    unmasked = x.unmasked
    a, a0, a1, alpha_on_0, alpha_on_1 = x.unwrapped

    with tf.name_scope("cache"):

        with tf.device(prot.crypto_producer.device_name):
            [a_cached], updater_cp = _cache_wrap_helper(prot, [a])

        with tf.device(prot.server_0.device_name):
            [a0_cached, alpha_on_0_cached], updater0 = _cache_wrap_helper(
                prot, [a0, alpha_on_0]
            )

        with tf.device(prot.server_1.device_name):
            [a1_cached, alpha_on_1_cached], updater1 = _cache_wrap_helper(
                prot, [a1, alpha_on_1]
            )

        updater = tf.group(updater_cp, updater0, updater1)
        unmasked_cached = prot.cache(unmasked)

    global_cache_updaters.append(updater)
    return PondCachedMaskedTensor(
        prot,
        unmasked_cached,
        a_cached,
        a0_cached,
        a1_cached,
        alpha_on_0_cached,
        alpha_on_1_cached,
        x.is_scaled,
        updater,
    )


#
# truncate
#


def _truncate_public(prot: Pond, x: PondPublicTensor) -> PondPublicTensor:
    assert isinstance(x, PondPublicTensor)

    base = prot.fixedpoint_config.scaling_base
    amount = prot.fixedpoint_config.precision_fractional
    x_on_0, x_on_1 = x.unwrapped

    with tf.name_scope("truncate"):

        with tf.device(prot.server_0.device_name):
            y_on_0 = x_on_0.truncate(amount, base)

        with tf.device(prot.server_1.device_name):
            y_on_1 = x_on_1.truncate(amount, base)

    return PondPublicTensor(prot, y_on_0, y_on_1, x.is_scaled)


def _truncate_private(prot: Pond, x: PondPrivateTensor) -> PondPrivateTensor:
    assert isinstance(x, PondPrivateTensor)

    if prot.fixedpoint_config.use_noninteractive_truncation:
        return _truncate_private_noninteractive(prot, x)
    else:
        return _truncate_private_interactive(prot, x)


def _truncate_private_noninteractive(
    prot: Pond, x: PondPrivateTensor
) -> PondPrivateTensor:
    assert isinstance(x, PondPrivateTensor)

    base = prot.fixedpoint_config.scaling_base
    amount = prot.fixedpoint_config.precision_fractional
    x0, x1 = x.unwrapped

    with tf.name_scope("truncate-ni"):

        with tf.device(prot.server_0.device_name):
            y0 = x0.truncate(amount, base)

        with tf.device(prot.server_1.device_name):
            y1 = 0 - (0 - x1).truncate(amount, base)

    return PondPrivateTensor(prot, y0, y1, x.is_scaled)


def _truncate_private_interactive(
    prot: Pond, a: PondPrivateTensor
) -> PondPrivateTensor:
    """ See protocol TruncPr (3.1) in "Secure Computation With Fixed-Point Numbers"
    by Octavian Catrina and Amitabh Saxena, FC'10. """

    with tf.name_scope("truncate-i"):

        scaling_factor = prot.fixedpoint_config.scaling_factor
        scaling_factor_inverse = inverse(
            prot.fixedpoint_config.scaling_factor, prot.tensor_factory.modulus
        )

        # we first rotate `a` to make sure reconstructed values fall into
        # a non-negative interval `[0, 2B)` for some bound B; this uses an
        # assumption that the values originally lie in `[-B, B)`, and will
        # leak private information otherwise

        bound = prot.fixedpoint_config.bound_double_precision
        b = a + bound

        # next step is for server0 to add a statistical mask to `b`, reveal
        # it to server1, and compute the lower part

        mask_bitlength = ceil(log2(bound)) + 1 + prot.fixedpoint_config.truncation_gap

        b0, b1 = b.unwrapped
        shape = a.shape

        with tf.device(prot.server_0.device_name):
            r = prot.tensor_factory.sample_bounded(shape, mask_bitlength)
            c0 = b0 + r

        with tf.device(prot.server_1.device_name):
            c1 = b1
            c_lower = prot._reconstruct(c0, c1) % scaling_factor

        # then use the lower part of the masked value to compute lower part
        # of original value

        with tf.device(prot.server_0.device_name):
            r_lower = r % scaling_factor
            a_lower0 = r_lower * -1

        with tf.device(prot.server_1.device_name):
            a_lower1 = c_lower

        # finally subtract and multiply by inverse

        a0, a1 = a.unwrapped

        with tf.device(prot.server_0.device_name):
            d0 = (a0 - a_lower0) * scaling_factor_inverse

        with tf.device(prot.server_1.device_name):
            d1 = (a1 - a_lower1) * scaling_factor_inverse

    return PondPrivateTensor(prot, d0, d1, a.is_scaled)


def _truncate_masked(prot: Pond, x: PondMaskedTensor) -> PondMaskedTensor:
    assert isinstance(x, PondMaskedTensor)
    return prot.truncate(x.unmasked)


#
# reveal helpers
#


def _reveal_private(prot, x):
    assert isinstance(x, PondPrivateTensor), type(x)

    with tf.name_scope("reveal"):

        x0, x1 = x.unwrapped

        with tf.device(prot.server_0.device_name):
            z_on_0 = x0 + x1

        with tf.device(prot.server_1.device_name):
            z_on_1 = x0 + x1

    return PondPublicTensor(prot, z_on_0, z_on_1, x.is_scaled)


def _reveal_masked(prot, x):
    assert isinstance(x, PondMaskedTensor), type(x)
    return prot.reveal(x.unmasked)


#
# add helpers
#


def _add_public_public(prot, x, y):
    assert isinstance(x, PondPublicTensor), type(x)
    assert isinstance(y, PondPublicTensor), type(y)
    assert x.is_scaled == y.is_scaled, "Cannot mix different encodings: {} {}".format(
        x.is_scaled, y.is_scaled
    )

    x_on_0, x_on_1 = x.unwrapped
    y_on_0, y_on_1 = y.unwrapped

    with tf.name_scope("add"):

        with tf.device(prot.server_0.device_name):
            z_on_0 = x_on_0 + y_on_0

        with tf.device(prot.server_1.device_name):
            z_on_1 = x_on_1 + y_on_1

    return PondPublicTensor(prot, z_on_0, z_on_1, x.is_scaled)


def _add_public_private(prot, x, y):
    assert isinstance(x, PondPublicTensor), type(x)
    assert isinstance(y, PondPrivateTensor), type(y)
    assert x.is_scaled == y.is_scaled, "Cannot mix different encodings: {} {}".format(
        x.is_scaled, y.is_scaled
    )

    x_on_0, _ = x.unwrapped
    y0, y1 = y.unwrapped

    with tf.name_scope("add"):

        with tf.device(prot.server_0.device_name):
            z0 = x_on_0 + y0

        with tf.device(prot.server_1.device_name):
            z1 = y1

    return PondPrivateTensor(prot, z0, z1, x.is_scaled)


def _add_public_masked(prot, x, y):
    assert isinstance(x, PondPublicTensor), type(x)
    assert isinstance(y, PondMaskedTensor), type(y)
    return prot.add(x, y.unmasked)


def _add_private_public(prot, x, y):
    assert isinstance(x, PondPrivateTensor), type(x)
    assert isinstance(y, PondPublicTensor), type(y)
    assert x.is_scaled == y.is_scaled, "Cannot mix different encodings: {} {}".format(
        x.is_scaled, y.is_scaled
    )

    x0, x1 = x.unwrapped
    y_on_0, _ = y.unwrapped

    with tf.name_scope("add"):

        with tf.device(prot.server_0.device_name):
            z0 = x0 + y_on_0

        with tf.device(prot.server_1.device_name):
            z1 = x1

    return PondPrivateTensor(prot, z0, z1, x.is_scaled)


def _add_private_private(prot, x, y):
    assert isinstance(x, PondPrivateTensor), type(x)
    assert isinstance(y, PondPrivateTensor), type(y)
    # TODO[Morten] fails due to use in masking in SecureNN; how do deal with this?
    # err = "Cannot mix different encodings: {} {}".format(x.is_scaled, y.is_scaled)
    # assert x.is_scaled == y.is_scaled, err

    x0, x1 = x.unwrapped
    y0, y1 = y.unwrapped

    with tf.name_scope("add"):

        with tf.device(prot.server_0.device_name):
            z0 = x0 + y0

        with tf.device(prot.server_1.device_name):
            z1 = x1 + y1

    return PondPrivateTensor(prot, z0, z1, x.is_scaled)


def _add_private_masked(prot, x, y):
    assert isinstance(x, PondPrivateTensor), type(x)
    assert isinstance(y, PondMaskedTensor), type(y)
    return prot.add(x, y.unmasked)


def _add_masked_public(prot, x, y):
    assert isinstance(x, PondMaskedTensor), type(x)
    assert isinstance(y, PondPublicTensor), type(y)
    return prot.add(x.unmasked, y)


def _add_masked_private(prot, x, y):
    assert isinstance(x, PondMaskedTensor), type(x)
    assert isinstance(y, PondPrivateTensor), type(y)
    return prot.add(x.unmasked, y)


def _add_masked_masked(prot, x, y):
    assert isinstance(x, PondMaskedTensor), type(x)
    assert isinstance(y, PondMaskedTensor), type(y)
    return prot.add(x.unmasked, y.unmasked)


#
# reduce_sum helpers
#


def _reduce_sum_public(
    prot: Pond,
    x: PondPublicTensor,
    axis: Optional[int] = None,
    keepdims: Optional[bool] = None,
) -> PondPublicTensor:

    x_on_0, x_on_1 = x.unwrapped

    with tf.name_scope("reduce_sum"):

        with tf.device(prot.server_0.device_name):
            y_on_0 = x_on_0.reduce_sum(axis, keepdims)

        with tf.device(prot.server_1.device_name):
            y_on_1 = x_on_1.reduce_sum(axis, keepdims)

    return PondPublicTensor(prot, y_on_0, y_on_1, x.is_scaled)


def _reduce_sum_private(
    prot: Pond,
    x: PondPrivateTensor,
    axis: Optional[int] = None,
    keepdims: Optional[bool] = None,
) -> PondPrivateTensor:

    x0, x1 = x.unwrapped

    with tf.name_scope("reduce_sum"):

        with tf.device(prot.server_0.device_name):
            y0 = x0.reduce_sum(axis, keepdims)

        with tf.device(prot.server_1.device_name):
            y1 = x1.reduce_sum(axis, keepdims)

    return PondPrivateTensor(prot, y0, y1, x.is_scaled)


def _reduce_sum_masked(
    prot: Pond,
    x: PondMaskedTensor,
    axis: Optional[int] = None,
    keepdims: Optional[bool] = None,
) -> PondPrivateTensor:
    return prot.reduce_sum(x.unmasked, axis, keepdims)


#
# cumsum helpers
#


def _cumsum_public(
    prot: Pond,
    x: PondPublicTensor,
    axis: Optional[int] = None,
    exclusive: Optional[bool] = None,
    reverse: Optional[bool] = None,
) -> PondPublicTensor:

    x_on_0, x_on_1 = x.unwrapped

    with tf.name_scope("cumsum"):

        with tf.device(prot.server_0.device_name):
            y_on_0 = x_on_0.cumsum(axis=axis, exclusive=exclusive, reverse=reverse)

        with tf.device(prot.server_1.device_name):
            y_on_1 = x_on_1.cumsum(axis=axis, exclusive=exclusive, reverse=reverse)

    return PondPublicTensor(prot, y_on_0, y_on_1, x.is_scaled)


def _cumsum_private(
    prot: Pond,
    x: PondPrivateTensor,
    axis: Optional[int] = None,
    exclusive: Optional[bool] = None,
    reverse: Optional[bool] = None,
) -> PondPrivateTensor:

    x0, x1 = x.unwrapped

    with tf.name_scope("cumsum"):

        with tf.device(prot.server_0.device_name):
            y0 = x0.cumsum(axis=axis, exclusive=exclusive, reverse=reverse)

        with tf.device(prot.server_1.device_name):
            y1 = x1.cumsum(axis=axis, exclusive=exclusive, reverse=reverse)

    return PondPrivateTensor(prot, y0, y1, x.is_scaled)


def _cumsum_masked(
    prot: Pond,
    x: PondMaskedTensor,
    axis: Optional[int] = None,
    exclusive: Optional[bool] = None,
    reverse: Optional[bool] = None,
) -> PondPrivateTensor:
    return prot.cumsum(x.unmasked, axis=axis, exclusive=exclusive, reverse=reverse)


#
# sub helpers
#


def _sub_public_public(prot, x, y):
    assert isinstance(x, PondPublicTensor), type(x)
    assert isinstance(y, PondPublicTensor), type(y)
    assert x.is_scaled == y.is_scaled, "Cannot mix different encodings: {} {}".format(
        x.is_scaled, y.is_scaled
    )

    x_on_0, x_on_1 = x.unwrapped
    y_on_0, y_on_1 = y.unwrapped

    with tf.name_scope("sub"):

        with tf.device(prot.server_0.device_name):
            z_on_0 = x_on_0 - y_on_0

        with tf.device(prot.server_1.device_name):
            z_on_1 = x_on_1 - y_on_1

    return PondPublicTensor(prot, z_on_0, z_on_1, x.is_scaled)


def _sub_public_private(prot, x, y):
    assert isinstance(x, PondPublicTensor), type(x)
    assert isinstance(y, PondPrivateTensor), type(y)
    assert x.is_scaled == y.is_scaled, "Cannot mix different encodings: {} {}".format(
        x.is_scaled, y.is_scaled
    )

    x_on_0, _ = x.unwrapped
    y0, y1 = y.unwrapped

    with tf.name_scope("sub"):

        with tf.device(prot.server_0.device_name):
            z0 = x_on_0 - y0

        with tf.device(prot.server_1.device_name):
            z1 = y1.negative()

    return PondPrivateTensor(prot, z0, z1, x.is_scaled)


def _sub_public_masked(prot, x, y):
    assert isinstance(x, PondPublicTensor), type(x)
    assert isinstance(y, PondMaskedTensor), type(y)
    return prot.sub(x, y.unmasked)


def _sub_private_public(prot, x, y):
    assert isinstance(x, PondPrivateTensor), type(x)
    assert isinstance(y, PondPublicTensor), type(y)
    assert x.is_scaled == y.is_scaled, "Cannot mix different encodings: {} {}".format(
        x.is_scaled, y.is_scaled
    )

    x0, x1 = x.unwrapped
    y_on_0, _ = y.unwrapped

    with tf.name_scope("sub"):

        with tf.device(prot.server_0.device_name):
            z0 = x0 - y_on_0

        with tf.device(prot.server_1.device_name):
            z1 = x1

    return PondPrivateTensor(prot, z0, z1, x.is_scaled)


def _sub_private_private(prot, x, y):
    assert isinstance(x, PondPrivateTensor), type(x)
    assert isinstance(y, PondPrivateTensor), type(y)
    assert x.is_scaled == y.is_scaled, "Cannot mix different encodings: {} {}".format(
        x.is_scaled, y.is_scaled
    )

    x0, x1 = x.unwrapped
    y0, y1 = y.unwrapped

    with tf.name_scope("sub"):

        with tf.device(prot.server_0.device_name):
            z0 = x0 - y0

        with tf.device(prot.server_1.device_name):
            z1 = x1 - y1

    return PondPrivateTensor(prot, z0, z1, x.is_scaled)


def _sub_private_masked(prot, x, y):
    assert isinstance(x, PondPrivateTensor), type(x)
    assert isinstance(y, PondMaskedTensor), type(y)
    return prot.sub(x, y.unmasked)


def _sub_masked_public(prot, x, y):
    assert isinstance(x, PondMaskedTensor), type(x)
    assert isinstance(y, PondPublicTensor), type(y)
    return prot.sub(x.unmasked, y)


def _sub_masked_private(prot, x, y):
    assert isinstance(x, PondMaskedTensor), type(x)
    assert isinstance(y, PondPrivateTensor), type(y)
    return prot.sub(x.unmasked, y)


def _sub_masked_masked(prot, x, y):
    assert isinstance(x, PondMaskedTensor), type(x)
    assert isinstance(y, PondMaskedTensor), type(y)
    return prot.sub(x.unmasked, y.unmasked)


#
# mul helpers
#


def _mul_public_public(prot, x, y):
    assert isinstance(x, PondPublicTensor), type(x)
    assert isinstance(y, PondPublicTensor), type(y)

    x_on_0, x_on_1 = x.unwrapped
    y_on_0, y_on_1 = y.unwrapped

    with tf.name_scope("mul"):

        with tf.device(prot.server_0.device_name):
            z_on_0 = x_on_0 * y_on_0

        with tf.device(prot.server_1.device_name):
            z_on_1 = x_on_1 * y_on_1

        z = PondPublicTensor(prot, z_on_0, z_on_1, x.is_scaled or y.is_scaled)
        z = prot.truncate(z) if x.is_scaled and y.is_scaled else z
        return z


def _mul_public_private(prot, x, y):
    assert isinstance(x, PondPublicTensor), type(x)
    assert isinstance(y, PondPrivateTensor), type(y)

    x_on_0, x_on_1 = x.unwrapped
    y0, y1 = y.unwrapped

    with tf.name_scope("mul"):

        with tf.device(prot.server_0.device_name):
            z0 = x_on_0 * y0

        with tf.device(prot.server_1.device_name):
            z1 = x_on_1 * y1

        z = PondPrivateTensor(prot, z0, z1, x.is_scaled or y.is_scaled)
        z = prot.truncate(z) if x.is_scaled and y.is_scaled else z
        return z


def _mul_public_masked(prot, x, y):
    assert isinstance(x, PondPublicTensor), type(x)
    assert isinstance(y, PondMaskedTensor), type(y)
    return prot.mul(x, y.unmasked)


def _mul_private_public(prot, x, y):
    assert isinstance(x, PondPrivateTensor), type(x)
    assert isinstance(y, PondPublicTensor), type(y)

    x0, x1 = x.unwrapped
    y_on_0, y_on_1 = y.unwrapped

    with tf.name_scope("mul"):

        with tf.device(prot.server_0.device_name):
            z0 = x0 * y_on_0

        with tf.device(prot.server_1.device_name):
            z1 = x1 * y_on_1

        z = PondPrivateTensor(prot, z0, z1, x.is_scaled or y.is_scaled)
        z = prot.truncate(z) if x.is_scaled and y.is_scaled else z
        return z


def _mul_private_private(prot, x, y):
    assert isinstance(x, PondPrivateTensor), type(x)
    assert isinstance(y, PondPrivateTensor), type(y)
    return prot.mul(prot.mask(x), prot.mask(y))


def _mul_private_masked(prot, x, y):
    assert isinstance(x, PondPrivateTensor), type(x)
    assert isinstance(y, PondMaskedTensor), type(y)
    return prot.mul(prot.mask(x), y)


def _mul_masked_public(prot, x, y):
    assert isinstance(x, PondMaskedTensor), type(x)
    assert isinstance(y, PondPublicTensor), type(y)
    return prot.mul(x.unmasked, y)


def _mul_masked_private(prot, x, y):
    assert isinstance(x, PondMaskedTensor), type(x)
    assert isinstance(y, PondPrivateTensor), type(y)
    return prot.mul(x, prot.mask(y))


def _mul_masked_masked(prot, x, y):
    assert isinstance(x, PondMaskedTensor), type(x)
    assert isinstance(y, PondMaskedTensor), type(y)

    a, a0, a1, alpha_on_0, alpha_on_1 = x.unwrapped
    b, b0, b1, beta_on_0, beta_on_1 = y.unwrapped

    with tf.name_scope("mul"):

        with tf.device(prot.crypto_producer.device_name):
            with tf.name_scope("triple"):
                ab = a * b
                ab0, ab1 = prot._share(ab)

        with tf.device(prot.server_0.device_name):
            with tf.name_scope("combine"):
                alpha = alpha_on_0
                beta = beta_on_0
                z0 = ab0 + (a0 * beta) + (alpha * b0) + (alpha * beta)

        with tf.device(prot.server_1.device_name):
            with tf.name_scope("combine"):
                alpha = alpha_on_1
                beta = beta_on_1
                z1 = ab1 + (a1 * beta) + (alpha * b1)

        z = PondPrivateTensor(prot, z0, z1, x.is_scaled or y.is_scaled)
        z = prot.truncate(z) if x.is_scaled and y.is_scaled else z
        return z


#
# square helpers
#


def _square_public(prot, x):
    assert isinstance(x, PondPublicTensor), type(x)

    x_on_0, x_on_1 = x.unwrapped

    with tf.name_scope("square"):

        with tf.device(prot.server_0.device_name):
            y_on_0 = x_on_0 * x_on_0

        with tf.device(prot.server_1.device_name):
            y_on_1 = x_on_1 * x_on_1

        y = PondPublicTensor(prot, y_on_0, y_on_1, x.is_scaled)
        y = prot.truncate(y) if y.is_scaled else y
        return y


def _square_private(prot, x):
    assert isinstance(x, PondPrivateTensor), type(x)
    return prot.square(prot.mask(x))


def _square_masked(prot, x):
    assert isinstance(x, PondMaskedTensor), type(x)

    a, a0, a1, alpha_on_0, alpha_on_1 = x.unwrapped

    with tf.name_scope("square"):

        with tf.device(prot.crypto_producer.device_name):
            with tf.name_scope("triple"):
                aa = a * a
                aa0, aa1 = prot._share(aa)

        with tf.device(prot.server_0.device_name):
            with tf.name_scope("combine"):
                alpha = alpha_on_0
                y0 = aa0 + (a0 * alpha) * 2 + (alpha * alpha)

        with tf.device(prot.server_1.device_name):
            with tf.name_scope("combine"):
                alpha = alpha_on_1
                y1 = aa1 + (a1 * alpha) * 2

        y = PondPrivateTensor(prot, y0, y1, x.is_scaled)
        y = prot.truncate(y) if y.is_scaled else y
        return y


#
# matmul helpers
#


def _matmul_public_public(
    prot, x: PondPublicTensor, y: PondPublicTensor
) -> PondPublicTensor:

    x_on_0, x_on_1 = x.unwrapped
    y_on_0, y_on_1 = y.unwrapped

    with tf.name_scope("matmul"):

        with tf.device(prot.server_0.device_name):
            z_on_0 = x_on_0.matmul(y_on_0)

        with tf.device(prot.server_1.device_name):
            z_on_1 = x_on_1.matmul(y_on_1)

        z = PondPublicTensor(prot, z_on_0, z_on_1, x.is_scaled or y.is_scaled)
        z = prot.truncate(z) if x.is_scaled and y.is_scaled else z
        return z


def _matmul_public_private(prot, x, y):
    assert isinstance(x, PondPublicTensor), type(x)
    assert isinstance(y, PondPrivateTensor), type(y)

    x_on_0, x_on_1 = x.unwrapped
    y0, y1 = y.unwrapped

    with tf.name_scope("matmul"):

        with tf.device(prot.server_0.device_name):
            z0 = x_on_0.matmul(y0)

        with tf.device(prot.server_1.device_name):
            z1 = x_on_1.matmul(y1)

        z = PondPrivateTensor(prot, z0, z1, x.is_scaled or y.is_scaled)
        z = prot.truncate(z) if x.is_scaled and y.is_scaled else z
        return z


def _matmul_public_masked(prot, x, y):
    assert isinstance(x, PondPublicTensor), type(x)
    assert isinstance(y, PondMaskedTensor), type(y)
    return prot.matmul(x, y.unmasked)


def _matmul_private_public(prot, x, y):
    assert isinstance(x, PondPrivateTensor), type(x)
    assert isinstance(y, PondPublicTensor), type(y)

    x0, x1 = x.unwrapped
    y_on_0, y_on_1 = y.unwrapped

    with tf.name_scope("matmul"):

        with tf.device(prot.server_0.device_name):
            z0 = x0.matmul(y_on_0)

        with tf.device(prot.server_0.device_name):
            z1 = x1.matmul(y_on_1)

        z = PondPrivateTensor(prot, z0, z1, x.is_scaled or y.is_scaled)
        z = prot.truncate(z) if x.is_scaled and y.is_scaled else z
        return z


def _matmul_private_private(prot, x, y):
    assert isinstance(x, PondPrivateTensor), type(x)
    assert isinstance(y, PondPrivateTensor), type(y)
    return prot.matmul(prot.mask(x), prot.mask(y))


def _matmul_private_masked(prot, x, y):
    assert isinstance(x, PondPrivateTensor), type(x)
    assert isinstance(y, PondMaskedTensor), type(y)
    return prot.matmul(prot.mask(x), y)


def _matmul_masked_public(prot, x, y):
    assert isinstance(x, PondMaskedTensor), type(x)
    assert isinstance(y, PondPublicTensor), type(y)
    return prot.matmul(x.unmasked, y)


def _matmul_masked_private(prot, x, y):
    assert isinstance(x, PondMaskedTensor), type(x)
    assert isinstance(y, PondPrivateTensor), type(y)
    return prot.matmul(x, prot.mask(y))


def _matmul_masked_masked(prot, x, y):
    assert isinstance(x, PondMaskedTensor), type(x)
    assert isinstance(y, PondMaskedTensor), type(y)

    a, a0, a1, alpha_on_0, alpha_on_1 = x.unwrapped
    b, b0, b1, beta_on_0, beta_on_1 = y.unwrapped

    with tf.name_scope("matmul"):

        with tf.device(prot.crypto_producer.device_name):
            with tf.name_scope("triple"):
                ab = a.matmul(b)
                ab0, ab1 = prot._share(ab)

        with tf.device(prot.server_0.device_name):
            with tf.name_scope("combine"):
                alpha = alpha_on_0
                beta = beta_on_0
                z0 = ab0 + a0.matmul(beta) + alpha.matmul(b0) + alpha.matmul(beta)

        with tf.device(prot.server_1.device_name):
            with tf.name_scope("combine"):
                alpha = alpha_on_1
                beta = beta_on_1
                z1 = ab1 + a1.matmul(beta) + alpha.matmul(b1)

        z = PondPrivateTensor(prot, z0, z1, x.is_scaled or y.is_scaled)
        z = prot.truncate(z) if x.is_scaled and y.is_scaled else z
        return z


#
# Conv helpers
#
# TODO[koen] create operations for all possible combinations


def _conv2d_public_public(prot, x, y, strides, padding):
    assert isinstance(x, PondPublicTensor), type(x)
    assert isinstance(y, PondPublicTensor), type(y)

    x_0, x_1 = x.unwrapped
    y_0, y_1 = y.unwrapped

    with tf.name_scope("conv2d"):

        with tf.device(prot.server_0.device_name):
            z0 = x_0.conv2d(y_0, strides, padding)

        with tf.device(prot.server_1.device_name):
            z1 = x_1.conv2d(y_1, strides, padding)

        z = PondPublicTensor(prot, z0, z1, x.is_scaled or y.is_scaled)
        z = prot.truncate(z) if x.is_scaled and y.is_scaled else z
        return z


def _conv2d_public_private(prot, x, y, strides, padding):
    raise NotImplementedError()


def _conv2d_public_masked(prot, x, y, strides, padding):
    raise NotImplementedError()


def _conv2d_private_public(prot, x, y, strides, padding):
    raise NotImplementedError()


def _conv2d_private_masked(prot, x, y, strides, padding):
    assert isinstance(x, PondPrivateTensor), type(x)
    assert isinstance(y, PondMaskedTensor), type(y)
    return prot.conv2d(prot.mask(x), y, strides, padding)


def _conv2d_private_private(prot, x, y, strides, padding):
    assert isinstance(x, PondPrivateTensor), type(x)
    assert isinstance(y, PondPrivateTensor), type(y)
    return prot.conv2d(prot.mask(x), prot.mask(y), strides, padding)


def _conv2d_masked_public(prot, x, y, strides, padding):
    raise NotImplementedError()


def _conv2d_masked_private(prot, x, y, strides, padding):
    assert isinstance(x, PondMaskedTensor), type(x)
    assert isinstance(y, PondPrivateTensor), type(y)
    return prot.conv2d(x, prot.mask(y), strides, padding)


def _conv2d_masked_masked(prot, x, y, strides, padding):
    assert isinstance(x, PondMaskedTensor), type(x)
    assert isinstance(y, PondMaskedTensor), type(y)

    a, a0, a1, alpha_on_0, alpha_on_1 = x.unwrapped
    b, b0, b1, beta_on_0, beta_on_1 = y.unwrapped

    with tf.name_scope("conv2d"):

        with tf.device(prot.crypto_producer.device_name):
            with tf.name_scope("triple"):
                a_conv2d_b = a.conv2d(b, strides, padding)
                a_conv2d_b0, a_conv2d_b1 = prot._share(a_conv2d_b)

        with tf.device(prot.server_0.device_name):
            with tf.name_scope("combine"):
                alpha = alpha_on_0
                beta = beta_on_0
                z0 = (
                    a_conv2d_b0
                    + a0.conv2d(beta, strides, padding)
                    + alpha.conv2d(b0, strides, padding)
                    + alpha.conv2d(beta, strides, padding)
                )

        with tf.device(prot.server_1.device_name):
            with tf.name_scope("combine"):
                alpha = alpha_on_1
                beta = beta_on_1
                z1 = (
                    a_conv2d_b1
                    + a1.conv2d(beta, strides, padding)
                    + alpha.conv2d(b1, strides, padding)
                )

        z = PondPrivateTensor(prot, z0, z1, x.is_scaled or y.is_scaled)
        z = prot.truncate(z) if x.is_scaled and y.is_scaled else z
        return z


#
# average pooling helpers
#


def _avgpool2d_core(
    prot: Pond,
    x: PondTensor,
    pool_size: Tuple[int, int],
    strides: Tuple[int, int],
    padding: str,
) -> Tuple[AbstractTensor, AbstractTensor, float]:
    x_on_0, x_on_1 = x.unwrapped
    _, _, H, W = x.shape
    scalar = 1 / (pool_size[0] * pool_size[1])
    siamese = pool_size == strides and pool_size[0] == pool_size[1]
    even = H.value % pool_size[0] == 0 and W.value % pool_size[1] == 0

    if siamese and even:
        pooler = _avgpool2d_reshape_reduce
    else:
        pooler = _avgpool2d_im2col_reduce

    with tf.device(prot.server_0.device_name):
        y_on_0 = pooler(x_on_0, pool_size, strides, padding)

    with tf.device(prot.server_1.device_name):
        y_on_1 = pooler(x_on_1, pool_size, strides, padding)

    return y_on_0, y_on_1, scalar


def _avgpool2d_reshape_reduce(
    x: AbstractTensor,
    pool_size: Tuple[int, int],
    strides: Tuple[int, int],
    padding: str,
) -> AbstractTensor:
    pool_height, pool_width = tf.Dimension(pool_size[0]), tf.Dimension(pool_size[1])
    N, C, H, W = x.shape
    x_reshaped = x.reshape(
        [N, C, H // pool_height, pool_height, W // pool_width, pool_width]
    )
    return x_reshaped.reduce_sum(axis=3).reduce_sum(axis=4)


def _avgpool2d_im2col_reduce(
    x: AbstractTensor,
    pool_size: Tuple[int, int],
    strides: Tuple[int, int],
    padding: str,
) -> AbstractTensor:
    batch, channels, height, width = x.shape
    pool_height, pool_width = pool_size

    if padding == "SAME":
        out_height = ceil(int(height) / strides[0])
        out_width = ceil(int(width) / strides[1])
    else:
        out_height = ceil((int(height) - pool_size[0] + 1) / strides[0])
        out_width = ceil((int(width) - pool_size[1] + 1) / strides[1])

    x_split = x.reshape((batch * channels, 1, height, width))
    x_cols = x_split.im2col(pool_height, pool_width, padding, strides[0])
    x_cols_sum = x_cols.reduce_sum(axis=0)
    out = x_cols_sum.reshape([out_height, out_width, batch, channels]).transpose(
        [2, 3, 0, 1]
    )
    return out


def _avgpool2d_public(
    prot: Pond,
    x: PondPublicTensor,
    pool_size: Tuple[int, int],
    strides: Tuple[int, int],
    padding: str,
) -> PondPublicTensor:

    with tf.name_scope("avgpool2d"):
        y_on_0, y_on_1, scalar = _avgpool2d_core(prot, x, pool_size, strides, padding)
        return PondPublicTensor(prot, y_on_0, y_on_1, x.is_scaled) * scalar


def _avgpool2d_private(
    prot: Pond,
    x: PondPrivateTensor,
    pool_size: Tuple[int, int],
    strides: Tuple[int, int],
    padding: str,
) -> PondPrivateTensor:

    with tf.name_scope("avgpool2d"):
        y_on_0, y_on_1, scalar = _avgpool2d_core(prot, x, pool_size, strides, padding)
        return PondPrivateTensor(prot, y_on_0, y_on_1, x.is_scaled) * scalar


def _avgpool2d_masked(
    prot: Pond,
    x: PondMaskedTensor,
    pool_size: Tuple[int, int],
    strides: Tuple[int, int],
    padding: str,
) -> PondPrivateTensor:

    with tf.name_scope("avgpool2d"):
        y_on_0, y_on_1, scalar = _avgpool2d_core(
            prot, x.unmasked, pool_size, strides, padding
        )
        return PondPrivateTensor(prot, y_on_0, y_on_1, x.is_scaled) * scalar


#
# batch_to_space_nd and space_to_batch_nd helpers
#


def _batch_to_space_nd_core(prot, tensor, block_shape, crops):
    tensor_on_0, tensor_on_1 = tensor.unwrapped

    with tf.device(prot.server_0.device_name):
        space_on_0 = tensor_on_0.batch_to_space_nd(block_shape, crops)

    with tf.device(prot.server_1.device_name):
        space_on_1 = tensor_on_1.batch_to_space_nd(block_shape, crops)

    return space_on_0, space_on_1


def _batch_to_space_nd_public(prot, tensor, block_shape, crops):

    with tf.name_scope("batch_to_space_nd"):
        space_on_0, space_on_1 = _batch_to_space_nd_core(prot, tensor, block_shape, crops)

    return PondPublicTensor(prot, space_on_0, space_on_1, tensor.is_scaled)


def _batch_to_space_nd_private(prot, tensor, block_shape, crops):

    with tf.name_scope("batch_to_space_nd"):
        space_on_0, space_on_1 = _batch_to_space_nd_core(prot, tensor, block_shape, crops)

    return PondPrivateTensor(prot, space_on_0, space_on_1, tensor.is_scaled)


def _batch_to_space_nd_masked(prot, tensor, block_shape, crops):

    with tf.name_scope("batch_to_space_nd"):
        space_on_0, space_on_1 = _batch_to_space_nd_core(prot, tensor.unmasked, block_shape, crops)

    return PondPrivateTensor(prot, space_on_0, space_on_1, tensor.is_scaled)


def _space_to_batch_nd_core(prot, tensor, block_shape, paddings):
    tensor_on_0, tensor_on_1 = tensor.unwrapped

    with tf.name_scope("space_to_batch_nd"):

        with tf.device(prot.server_0.device_name):
            batch_on_0 = tensor_on_0.space_to_batch_nd(block_shape, paddings)

        with tf.device(prot.server_1.device_name):
            batch_on_1 = tensor_on_1.space_to_batch_nd(block_shape, paddings)

    return batch_on_0, batch_on_1


def _space_to_batch_nd_public(prot, tensor, block_shape, paddings):

    with tf.name_scope("space_to_batch_nd"):
        batch_on_0, batch_on_1 = _space_to_batch_nd_core(prot, tensor, block_shape, paddings)

    return PondPublicTensor(prot, batch_on_0, batch_on_1, tensor.is_scaled)


def _space_to_batch_nd_private(prot, tensor, block_shape, paddings):

    with tf.name_scope("space_to_batch_nd"):
        batch_on_0, batch_on_1 = _space_to_batch_nd_core(prot, tensor, block_shape, paddings)

    return PondPrivateTensor(prot, batch_on_0, batch_on_1, tensor.is_scaled)


def _space_to_batch_nd_masked(prot, tensor, block_shape, paddings):

    with tf.name_scope("space_to_batch_nd"):
        batch_on_0, batch_on_1 = _space_to_batch_nd_core(prot,
                                                         tensor.unmasked,
                                                         block_shape,
                                                         paddings)

    return PondPrivateTensor(prot, batch_on_0, batch_on_1, tensor.is_scaled)


#
# indexing helpers
#


def _indexer_public(
    prot: Pond, tensor: PondPublicTensor, slice: Union[Slice, Ellipse]
) -> "PondPublicTensor":

    with tf.name_scope("index"):

        with tf.device(prot.server_0.device_name):
            v_on_0 = tensor.value_on_0[slice]

        with tf.device(prot.server_1.device_name):
            v_on_1 = tensor.value_on_1[slice]

        return PondPublicTensor(prot, v_on_0, v_on_1, tensor.is_scaled)


def _indexer_private(
    prot: Pond, tensor: PondPrivateTensor, slice: Union[Slice, Ellipse]
) -> "PondPrivateTensor":

    with tf.name_scope("index"):

        with tf.device(prot.server_0.device_name):
            s0 = tensor.share0[slice]

        with tf.device(prot.server_1.device_name):
            s1 = tensor.share1[slice]

    return PondPrivateTensor(prot, s0, s1, tensor.is_scaled)


def _indexer_masked(
    prot: Pond, tensor: PondMaskedTensor, slice: Union[Slice, Ellipse]
) -> "PondMaskedTensor":

    with tf.name_scope("index"):

        with tf.device(prot.crypto_producer.device_name):
            a = tensor.a[slice]

        with tf.device(prot.server_0.device_name):
            a0 = tensor.a0[slice]
            alpha_on_0 = tensor.alpha_on_0[slice]

        with tf.device(prot.server_1.device_name):
            a1 = tensor.a1[slice]
            alpha_on_1 = tensor.alpha_on_1[slice]

        return PondMaskedTensor(
            prot,
            tensor.unmasked[slice],
            a,
            a0,
            a1,
            alpha_on_0,
            alpha_on_1,
            tensor.is_scaled,
        )


#
# transpose helpers
#


def _transpose_public(prot, x, perm=None):
    assert isinstance(x, PondPublicTensor)

    x_on_0, x_on_1 = x.unwrapped

    with tf.name_scope("transpose"):

        with tf.device(prot.server_0.device_name):
            x_on_0_t = x_on_0.transpose(perm=perm)

        with tf.device(prot.server_1.device_name):
            x_on_1_t = x_on_1.transpose(perm=perm)

        return PondPublicTensor(prot, x_on_0_t, x_on_1_t, x.is_scaled)


def _transpose_private(prot, x, perm=None):
    assert isinstance(x, PondPrivateTensor)

    x0, x1 = x.unwrapped

    with tf.name_scope("transpose"):

        with tf.device(prot.server_0.device_name):
            x0_t = x0.transpose(perm=perm)

        with tf.device(prot.server_1.device_name):
            x1_t = x1.transpose(perm=perm)

        return PondPrivateTensor(prot, x0_t, x1_t, x.is_scaled)


def _transpose_masked(prot, x, perm=None):
    assert isinstance(x, PondMaskedTensor)

    a, a0, a1, alpha_on_0, alpha_on_1 = x.unwrapped

    with tf.name_scope("transpose"):

        with tf.device(prot.crypto_producer.device_name):
            a_t = a.transpose(perm=perm)

        with tf.device(prot.server_0.device_name):
            a0_t = a0.transpose(perm=perm)
            alpha_on_0_t = alpha_on_0.transpose(perm=perm)

        with tf.device(prot.server_1.device_name):
            a1_t = a1.transpose(perm=perm)
            alpha_on_1_t = alpha_on_1.transpose(perm=perm)

        return PondMaskedTensor(
            prot,
            prot.transpose(x.unmasked, perm=perm),
            a_t,
            a0_t,
            a1_t,
            alpha_on_0_t,
            alpha_on_1_t,
            x.is_scaled,
        )


#
# strided slice helpers
#


def _strided_slice_public(prot, x: PondPublicTensor, args: Any, kwargs: Any):
    assert isinstance(x, PondPublicTensor)

    x_on_0, x_on_1 = x.unwrapped

    with tf.name_scope("strided_slice"):

        with tf.device(prot.server_0.device_name):
            x_on_0_slice = x_on_0.strided_slice(args, kwargs)

        with tf.device(prot.server_1.device_name):
            x_on_1_slice = x_on_1.strided_slice(args, kwargs)

        return PondPublicTensor(prot, x_on_0_slice, x_on_1_slice, x.is_scaled)


def _strided_slice_private(prot, x: PondPrivateTensor, args: Any, kwargs: Any):
    assert isinstance(x, PondPrivateTensor)

    x0, x1 = x.unwrapped

    with tf.name_scope("strided_slice"):

        with tf.device(prot.server_0.device_name):
            x0_slice = x0.strided_slice(args, kwargs)

        with tf.device(prot.server_1.device_name):
            x1_slice = x1.strided_slice(args, kwargs)

        return PondPrivateTensor(prot, x0_slice, x1_slice, x.is_scaled)


def _strided_slice_masked(prot, x: PondMaskedTensor, args: Any, kwargs: Any):
    assert isinstance(x, PondMaskedTensor)

    a, a0, a1, alpha_on_0, alpha_on_1 = x.unwrapped

    with tf.name_scope("strided_slice"):

        with tf.device(prot.crypto_producer.device_name):
            a_slice = a.strided_slice(args, kwargs)

        with tf.device(prot.server_0.device_name):
            a0_slice = a0.strided_slice(args, kwargs)
            alpha_on_0_slice = alpha_on_0.strided_slice(args, kwargs)

        with tf.device(prot.server_1.device_name):
            a1_slice = a1.strided_slice(args, kwargs)
            alpha_on_1_slice = alpha_on_1.strided_slice(args, kwargs)

        return PondMaskedTensor(
            prot,
            prot.strided_slice(x.unmasked, args, kwargs),
            a_slice,
            a0_slice,
            a1_slice,
            alpha_on_0_slice,
            alpha_on_1_slice,
            x.is_scaled,
        )


#
# split helpers
#


def _split_public(
    prot: Pond, x: PondPublicTensor, num_split: int, axis: int = 0
) -> List[PondPublicTensor]:

    x_on_0, x_on_1 = x.unwrapped

    with tf.name_scope("split"):

        with tf.device(prot.server_0.device_name):
            ys_on_0 = x_on_0.split(num_split, axis=axis)

        with tf.device(prot.server_1.device_name):
            ys_on_1 = x_on_1.split(num_split, axis=axis)

        return [
            PondPublicTensor(prot, y_on_0, y_on_1, x.is_scaled)
            for y_on_0, y_on_1 in zip(ys_on_0, ys_on_1)
        ]


def _split_private(
    prot: Pond, x: PondPrivateTensor, num_split: int, axis: int = 0
) -> List[PondPrivateTensor]:

    x0, x1 = x.unwrapped

    with tf.name_scope("split"):

        with tf.device(prot.server_0.device_name):
            ys0 = x0.split(num_split, axis=axis)

        with tf.device(prot.server_1.device_name):
            ys1 = x1.split(num_split, axis=axis)

        return [
            PondPrivateTensor(prot, y0, y1, x.is_scaled) for y0, y1 in zip(ys0, ys1)
        ]


def _split_masked(
    prot: Pond, x: PondMaskedTensor, num_split: int, axis: int = 0
) -> List[PondMaskedTensor]:

    a, a0, a1, alpha_on_0, alpha_on_1 = x.unwrapped

    with tf.name_scope("split"):

        with tf.device(prot.crypto_producer.device_name):
            bs = a.split(num_split, axis=axis)

        with tf.device(prot.server_0.device_name):
            bs0 = a0.split(num_split, axis=axis)
            betas_on_0 = alpha_on_0.split(num_split, axis=axis)

        with tf.device(prot.server_1.device_name):
            bs1 = a1.split(num_split, axis=axis)
            betas_on_1 = alpha_on_1.split(num_split, axis=axis)

            ys = prot.split(x.unmasked, num_split, axis=axis)

        return [
            PondMaskedTensor(prot, y, b, b0, b1, beta_on_0, beta_on_1, x.is_scaled)
            for y, b, b0, b1, beta_on_0, beta_on_1 in zip(
                ys, bs, bs0, bs1, betas_on_0, betas_on_1
            )
        ]


#
# stack helpers
#


def _stack_public(
    prot: Pond, xs: List[PondPublicTensor], axis: int = 0
) -> PondPublicTensor:
    assert all(x.is_scaled for x in xs) or all(not x.is_scaled for x in xs)

    factory = xs[0].backing_dtype
    is_scaled = xs[0].is_scaled
    xs_on_0, xs_on_1 = zip(*(x.unwrapped for x in xs))

    with tf.name_scope("stack"):

        with tf.device(prot.server_0.device_name):
            x_on_0_stacked = factory.stack(xs_on_0, axis=axis)

        with tf.device(prot.server_1.device_name):
            x_on_1_stacked = factory.stack(xs_on_1, axis=axis)

        return PondPublicTensor(prot, x_on_0_stacked, x_on_1_stacked, is_scaled)


def _stack_private(
    prot: Pond, xs: List[PondPrivateTensor], axis: int = 0
) -> PondPrivateTensor:
    assert all(x.is_scaled for x in xs) or all(not x.is_scaled for x in xs)

    factory = xs[0].backing_dtype
    is_scaled = xs[0].is_scaled
    xs0, xs1 = zip(*(x.unwrapped for x in xs))

    with tf.name_scope("stack"):

        with tf.device(prot.server_0.device_name):
            x0_stacked = factory.stack(xs0, axis=axis)

        with tf.device(prot.server_1.device_name):
            x1_stacked = factory.stack(xs1, axis=axis)

        return PondPrivateTensor(prot, x0_stacked, x1_stacked, is_scaled)


def _stack_masked(
    prot: Pond, xs: List[PondMaskedTensor], axis: int = 0
) -> PondMaskedTensor:
    assert all(x.is_scaled for x in xs) or all(not x.is_scaled for x in xs)

    factory = xs[0].backing_dtype
    is_scaled = xs[0].is_scaled
    a, a0, a1, alpha_on_0, alpha_on_1 = zip(*(x.unwrapped for x in xs))

    with tf.name_scope("stack"):

        with tf.device(prot.crypto_producer.device_name):
            a_stacked = factory.stack(a, axis=axis)

        with tf.device(prot.server_0.device_name):
            a0_stacked = factory.stack(a0, axis=axis)
            alpha_on_0_stacked = factory.stack(alpha_on_0, axis=axis)

        with tf.device(prot.server_1.device_name):
            a1_stacked = factory.stack(a1, axis=axis)
            alpha_on_1_stacked = factory.stack(alpha_on_1, axis=axis)

        return PondMaskedTensor(
            prot,
            prot.stack([x.unmasked for x in xs], axis=axis),
            a_stacked,
            a0_stacked,
            a1_stacked,
            alpha_on_0_stacked,
            alpha_on_1_stacked,
            is_scaled,
        )


#
# concat helpers
#


def _concat_public(
    prot: Pond, xs: List[PondPublicTensor], axis: int
) -> PondPublicTensor:
    assert all(x.is_scaled for x in xs) or all(not x.is_scaled for x in xs)

    factory = xs[0].backing_dtype
    is_scaled = xs[0].is_scaled
    xs_on_0, xs_on_1 = zip(*(x.unwrapped for x in xs))

    with tf.name_scope("concat"):

        with tf.device(prot.server_0.device_name):
            x_on_0_concat = factory.concat(xs_on_0, axis=axis)

        with tf.device(prot.server_1.device_name):
            x_on_1_concat = factory.concat(xs_on_1, axis=axis)

        return PondPublicTensor(prot, x_on_0_concat, x_on_1_concat, is_scaled)


def _concat_private(
    prot: Pond, xs: List[PondPrivateTensor], axis: int
) -> PondPrivateTensor:
    assert all(x.is_scaled for x in xs) or all(not x.is_scaled for x in xs)

    factory = xs[0].backing_dtype
    is_scaled = xs[0].is_scaled
    xs0, xs1 = zip(*(x.unwrapped for x in xs))

    with tf.name_scope("concat"):

        with tf.device(prot.server_0.device_name):
            x0_concat = factory.concat(xs0, axis=axis)

        with tf.device(prot.server_1.device_name):
            x1_concat = factory.concat(xs1, axis=axis)

        return PondPrivateTensor(prot, x0_concat, x1_concat, is_scaled)


def _concat_masked(
    prot: Pond, xs: List[PondMaskedTensor], axis: int
) -> PondMaskedTensor:
    assert all(x.is_scaled for x in xs) or all(not x.is_scaled for x in xs)

    factory = xs[0].backing_dtype
    is_scaled = xs[0].is_scaled
    a, a0, a1, alpha_on_0, alpha_on_1 = zip(*(x.unwrapped for x in xs))

    with tf.name_scope("concat"):

        with tf.device(prot.crypto_producer.device_name):
            a_concat = factory.concat(a, axis=axis)

        with tf.device(prot.server_0.device_name):
            a0_concat = factory.concat(a0, axis=axis)
            alpha_on_0_concat = factory.concat(alpha_on_0, axis=axis)

        with tf.device(prot.server_1.device_name):
            a1_concat = factory.concat(a1, axis=axis)
            alpha_on_1_concat = factory.concat(alpha_on_1, axis=axis)

        return PondMaskedTensor(
            prot,
            prot.concat([x.unmasked for x in xs], axis=axis),
            a_concat,
            a0_concat,
            a1_concat,
            alpha_on_0_concat,
            alpha_on_1_concat,
            is_scaled,
        )


#
# mask helpers
#


def _mask_private(prot: Pond, x: PondPrivateTensor) -> PondMaskedTensor:
    assert isinstance(x, PondPrivateTensor)

    x0, x1 = x.unwrapped

    with tf.name_scope("mask"):

        with tf.device(prot.crypto_producer.device_name):
            a0 = x.backing_dtype.sample_uniform(x.shape)
            a1 = x.backing_dtype.sample_uniform(x.shape)
            a = a0 + a1

        with tf.device(prot.server_0.device_name):
            alpha0 = x0 - a0

        with tf.device(prot.server_1.device_name):
            alpha1 = x1 - a1

        with tf.device(prot.server_0.device_name):
            alpha_on_0 = alpha0 + alpha1

        with tf.device(prot.server_1.device_name):
            alpha_on_1 = alpha0 + alpha1

        return PondMaskedTensor(prot, x, a, a0, a1, alpha_on_0, alpha_on_1, x.is_scaled)


#
# reshape helpers
#


def _reshape_public(
    prot: Pond, x: PondPublicTensor, shape: List[int]
) -> PondPublicTensor:
    assert isinstance(x, PondPublicTensor)

    x_on_0, x_on_1 = x.unwrapped

    with tf.name_scope("reshape"):

        with tf.device(prot.server_0.device_name):
            x_on_0_reshaped = x_on_0.reshape(shape)

        with tf.device(prot.server_1.device_name):
            x_on_1_reshaped = x_on_1.reshape(shape)

        return PondPublicTensor(prot, x_on_0_reshaped, x_on_1_reshaped, x.is_scaled)


def _reshape_private(
    prot: Pond, x: PondPrivateTensor, shape: List[int]
) -> PondPrivateTensor:
    assert isinstance(x, PondPrivateTensor)

    x0, x1 = x.unwrapped

    with tf.name_scope("reshape"):

        with tf.device(prot.server_0.device_name):
            x0_reshaped = x0.reshape(shape)

        with tf.device(prot.server_1.device_name):
            x1_reshaped = x1.reshape(shape)

        return PondPrivateTensor(prot, x0_reshaped, x1_reshaped, x.is_scaled)


def _reshape_masked(
    prot: Pond, x: PondMaskedTensor, shape: List[int]
) -> PondMaskedTensor:
    assert isinstance(x, PondMaskedTensor)
    a, a0, a1, alpha_on_0, alpha_on_1 = x.unwrapped

    with tf.name_scope("reshape"):

        with tf.device(prot.crypto_producer.device_name):
            a_reshaped = a.reshape(shape)

        with tf.device(prot.server_0.device_name):
            a0_reshaped = a0.reshape(shape)
            alpha_on_0_reshaped = alpha_on_0.reshape(shape)

        with tf.device(prot.server_1.device_name):
            a1_reshaped = a1.reshape(shape)
            alpha_on_1_reshaped = alpha_on_1.reshape(shape)

        return PondMaskedTensor(
            prot,
            prot.reshape(x.unmasked, shape),
            a_reshaped,
            a0_reshaped,
            a1_reshaped,
            alpha_on_0_reshaped,
            alpha_on_1_reshaped,
            x.is_scaled,
        )

#
# negative helpers
#


def _negative_public(
    prot: Pond, x: PondPublicTensor
) -> PondPublicTensor:
    assert isinstance(x, PondPublicTensor)

    x_on_0, x_on_1 = x.unwrapped

    with tf.name_scope("negative"):

        with tf.device(prot.server_0.device_name):
            x_on_0_negative = x_on_0.negative()

        with tf.device(prot.server_1.device_name):
            x_on_1_negative = x_on_1.negative()

        return PondPublicTensor(prot, x_on_0_negative, x_on_1_negative, x.is_scaled)


def _negative_private(
    prot: Pond, x: PondPrivateTensor
) -> PondPrivateTensor:
    assert isinstance(x, PondPrivateTensor)

    x0, x1 = x.unwrapped

    with tf.name_scope("negative"):

        with tf.device(prot.server_0.device_name):
            x0_negative = x0.negative()

        with tf.device(prot.server_1.device_name):
            x1_negative = x1.negative()

        return PondPrivateTensor(prot, x0_negative, x1_negative, x.is_scaled)


def _negative_masked(
    prot: Pond, x: PondMaskedTensor
) -> PondMaskedTensor:
    assert isinstance(x, PondMaskedTensor)
    a, a0, a1, alpha_on_0, alpha_on_1 = x.unwrapped

    with tf.name_scope("negative"):

        with tf.device(prot.crypto_producer.device_name):
            a_negative = a.negative()

        with tf.device(prot.server_0.device_name):
            a0_negative = a0.negative()
            alpha_on_0_negative = alpha_on_0.negative()

        with tf.device(prot.server_1.device_name):
            a1_negative = a1.negative()
            alpha_on_1_negative = alpha_on_1.negative()

        return PondMaskedTensor(
            prot,
            prot.negative(x.unmasked),
            a_negative,
            a0_negative,
            a1_negative,
            alpha_on_0_negative,
            alpha_on_1_negative,
            x.is_scaled,
        )


#
# expand dims helpers
#


def _expand_dims_public(
    prot: Pond, x: PondPublicTensor, axis: Optional[int] = None
) -> PondPublicTensor:
    assert isinstance(x, PondPublicTensor)

    x_on_0, x_on_1 = x.unwrapped

    with tf.name_scope("expand"):

        with tf.device(prot.server_0.device_name):
            x_on_0_e = x_on_0.expand_dims(axis=axis)

        with tf.device(prot.server_1.device_name):
            x_on_1_e = x_on_1.expand_dims(axis=axis)

        return PondPublicTensor(prot, x_on_0_e, x_on_1_e, x.is_scaled)


def _expand_dims_private(
    prot: Pond, x: PondPrivateTensor, axis: Optional[int] = None
) -> PondPrivateTensor:
    assert isinstance(x, PondPrivateTensor)

    x0, x1 = x.unwrapped

    with tf.name_scope("expand"):

        with tf.device(prot.server_0.device_name):
            x0_e = x0.expand_dims(axis=axis)

        with tf.device(prot.server_1.device_name):
            x1_e = x1.expand_dims(axis=axis)

        return PondPrivateTensor(prot, x0_e, x1_e, x.is_scaled)


def _expand_dims_masked(
    prot: Pond, x: PondMaskedTensor, axis: Optional[int] = None
) -> PondMaskedTensor:
    assert isinstance(x, PondMaskedTensor)
    a, a0, a1, alpha_on_0, alpha_on_1 = x.unwrapped

    with tf.name_scope("expand"):

        with tf.device(prot.crypto_producer.device_name):
            a_e = a.expand_dims(axis=axis)

        with tf.device(prot.server_0.device_name):
            a0_e = a0.expand_dims(axis=axis)
            alpha_on_0_e = alpha_on_0.expand_dims(axis=axis)

        with tf.device(prot.server_1.device_name):
            a1_e = a1.expand_dims(axis=axis)
            alpha_on_1_e = alpha_on_1.expand_dims(axis=axis)

        return PondMaskedTensor(
            prot,
            prot.expand_dims(x.unmasked, axis=axis),
            a_e,
            a0_e,
            a1_e,
            alpha_on_0_e,
            alpha_on_1_e,
            x.is_scaled,
        )


#
# squeeze helpers
#


def _squeeze_public(
    prot: Pond, x: PondPublicTensor, axis: Optional[int] = None
) -> PondPublicTensor:
    assert isinstance(x, PondPublicTensor)

    x_on_0, x_on_1 = x.unwrapped

    with tf.name_scope("squeeze"):

        with tf.device(prot.server_0.device_name):
            x_on_0_squeezed = x_on_0.squeeze(axis)

        with tf.device(prot.server_1.device_name):
            x_on_1_squeezed = x_on_1.squeeze(axis)

        return PondPublicTensor(prot, x_on_0_squeezed, x_on_1_squeezed, x.is_scaled)


def _squeeze_private(
    prot: Pond, x: PondPrivateTensor, axis: Optional[int] = None
) -> PondPrivateTensor:
    assert isinstance(x, PondPrivateTensor)

    x0, x1 = x.unwrapped

    with tf.name_scope("squeeze"):

        with tf.device(prot.server_0.device_name):
            x0_squeezed = x0.squeeze(axis)

        with tf.device(prot.server_1.device_name):
            x1_squeezed = x1.squeeze(axis)

        return PondPrivateTensor(prot, x0_squeezed, x1_squeezed, x.is_scaled)


def _squeeze_masked(
    prot: Pond, x: PondMaskedTensor, axis: Optional[int] = None
) -> PondMaskedTensor:
    assert isinstance(x, PondMaskedTensor)
    a, a0, a1, alpha_on_0, alpha_on_1 = x.unwrapped

    with tf.name_scope("squeeze"):

        with tf.device(prot.crypto_producer.device_name):
            a_squeezed = a.squeeze(axis)

        with tf.device(prot.server_0.device_name):
            a0_squeezed = a0.squeeze(axis)
            alpha_on_0_squeezed = alpha_on_0.squeeze(axis)

        with tf.device(prot.server_1.device_name):
            a1_squeezed = a1.squeeze(axis)
            alpha_on_1_squeezed = alpha_on_1.squeeze(axis)

        return PondMaskedTensor(
            prot,
            prot.squeeze(x.unmasked),
            a_squeezed,
            a0_squeezed,
            a1_squeezed,
            alpha_on_0_squeezed,
            alpha_on_1_squeezed,
            x.is_scaled,
        )


#
# equal helpers
#


def _equal_public_public(
    prot: Pond, x: PondPublicTensor, y: PondPublicTensor
) -> PondPublicTensor:

    x_on_0, x_on_1 = x.unwrapped
    y_on_0, y_on_1 = y.unwrapped

    with tf.name_scope("equal"):

        with tf.device(prot.server_0.device_name):
            z_on_0 = x_on_0.equal(y_on_0)

        with tf.device(prot.server_0.device_name):
            z_on_1 = x_on_1.equal(y_on_1)

        return PondPublicTensor(prot, z_on_0, z_on_1, False)


#
# zeros helpers
#


def _zeros_private(
    prot,
    shape,
    apply_scaling: bool = True,
    name: Optional[str] = None,
    factory: Optional[AbstractFactory] = None
) -> 'PondPrivateTensor':

    zeros_array = np.zeros(shape)

    factory = factory or prot.tensor_factory

    with tf.name_scope('private-zeros{}'.format('-' + name if name else '')):

        v = factory.tensor(prot._encode(zeros_array, apply_scaling))
        v0, v1 = prot._share(v)

        with tf.device(prot.server_0.device_name):
            x0 = factory.variable(v0)

        with tf.device(prot.server_1.device_name):
            x1 = factory.variable(v1)

    x = PondPrivateTensor(prot, x0, x1, apply_scaling)
    return x


def _zeros_public(
    prot,
    shape,
    apply_scaling: bool = True,
    name: Optional[str] = None,
    factory: Optional[AbstractFactory] = None
) -> 'PondPublicTensor':

    zeros_array = np.zeros(shape)

    factory = factory or prot.tensor_factory

    with tf.name_scope('private-zeros{}'.format('-' + name if name else '')):

        v = factory.tensor(prot._encode(zeros_array, apply_scaling))

        with tf.device(prot.server_0.device_name):
            x0 = factory.variable(v)

        with tf.device(prot.server_1.device_name):
            x1 = factory.variable(v)

    x = PondPublicTensor(prot, x0, x1, apply_scaling)
    return x
