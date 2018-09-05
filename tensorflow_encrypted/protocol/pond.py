from __future__ import absolute_import
from typing import Tuple, Dict, List, Union, Optional, Any, NewType

import math
import numpy as np
import tensorflow as tf

from ..tensor.int100 import (
    Int100Tensor as BackingTensor,
    Int100Constant as BackingConstant,
    Int100Variable as BackingVariable,
    Int100Placeholder as BackingPlaceholder,
    stack,
)
from ..tensor.helpers import (
    log2,
    gcd,
    inverse
)
from ..io import InputProvider, OutputReceiver
from ..player import Player
from .protocol import Protocol, _global_cache_updators

TFEData = Union[np.ndarray, tf.Tensor]
TFEVariable = Union['PondPublicVariable', 'PondPrivateVariable', tf.Variable]
TFEPublicTensor = NewType('TFEPublicTensor', 'PondPublicTensor')
TFETensor = Union[TFEPublicTensor, 'PondPrivateTensor']

# the assumption in encoding/decoding is that encoded numbers will fit into signed int32
BITPRECISION_INTEGRAL = 14
BITPRECISION_FRACTIONAL = 16
TRUNCATION_GAP = 20
BOUND = 2**30  # bound on magnitude of encoded numbers: x in [-BOUND, +BOUND)

# modulus
M = BackingTensor.modulus
# truncation factor for going from double to single precision
K = 2 ** BITPRECISION_FRACTIONAL

assert log2(BITPRECISION_INTEGRAL + BITPRECISION_FRACTIONAL) <= log2(BOUND)
assert log2(M) >= 2 * (BITPRECISION_INTEGRAL + BITPRECISION_FRACTIONAL) + log2(1024) + TRUNCATION_GAP
assert gcd(K, M) == 1

_nodes: Dict = dict()
_initializers: List = list()


class Pond(Protocol):

    def __init__(self, server_0: Player, server_1: Player, crypto_producer: Player) -> None:
        self.server_0 = server_0
        self.server_1 = server_1
        self.crypto_producer = crypto_producer

    def define_constant(self, value: np.ndarray, apply_scaling: bool = True, name: Optional[str] = None) -> 'PondConstant':
        assert isinstance(value, (np.ndarray,)), type(value)

        v: BackingTensor = _encode(value, apply_scaling)

        with tf.name_scope('constant{}'.format('-' + name if name else '')):

            with tf.device(self.server_0.device_name):
                x_on_0 = BackingConstant.from_int100(v)

            with tf.device(self.server_1.device_name):
                x_on_1 = BackingConstant.from_int100(v)

        x = PondConstant(self, x_on_0, x_on_1)
        return x

    def define_public_placeholder(self, shape, name=None):

        with tf.name_scope('public-placeholder{}'.format('-' + name if name else '')):

            with tf.device(self.server_0.device_name):
                x_on_0 = BackingPlaceholder(shape)

            with tf.device(self.server_1.device_name):
                x_on_1 = BackingPlaceholder(shape)

        return PondPublicPlaceholder(self, x_on_0, x_on_1)

    def define_private_placeholder(self, shape, name=None):

        pl = BackingPlaceholder(shape)
        v0, v1 = _share(pl)
        assert type(v0) is BackingTensor, type(v0)
        assert type(v1) is BackingTensor, type(v1)

        with tf.name_scope('private-placeholder{}'.format('-' + name if name else '')):

            with tf.device(self.server_0.device_name):
                x0 = BackingTensor.from_decomposed(v0.backing)

            with tf.device(self.server_1.device_name):
                x1 = BackingTensor.from_decomposed(v1.backing)

        return PondPrivatePlaceholder(self, pl, x0, x1)

    def define_public_variable(
        self,
        initial_value,
        apply_scaling: bool=True,
        name: Optional[str]=None
    ) -> 'PondPublicVariable':
        assert isinstance(initial_value, (np.ndarray, tf.Tensor, PondPublicTensor)), type(initial_value)

        with tf.name_scope('public-var{}'.format('-' + name if name else '')):

            if isinstance(initial_value, (np.ndarray, tf.Tensor)):
                v: BackingTensor = _encode(initial_value, apply_scaling)
                v_on_0, v_on_1 = v, v

            elif isinstance(initial_value, PondPublicTensor):
                v_on_0, v_on_1 = initial_value.unwrapped

            else:
                raise TypeError(
                    "Don't know how to turn {} into public variable".format(type(initial_value)))

            with tf.device(self.server_0.device_name):
                x_on_0 = BackingVariable.from_int100(v_on_0)

            with tf.device(self.server_1.device_name):
                x_on_1 = BackingVariable.from_int100(v_on_1)

        x = PondPublicVariable(self, x_on_0, x_on_1)
        _initializers.append(x.initializer)
        return x

    def define_private_variable(
        self,
        initial_value,
        apply_scaling: bool=True,
        name: Optional[str]=None
    ) -> 'PondPrivateVariable':
        assert isinstance(initial_value, (np.ndarray, tf.Tensor, PondPublicTensor,
                                          PondPrivateTensor)), type(initial_value)

        with tf.name_scope('private-var{}'.format('-' + name if name else '')):

            if isinstance(initial_value, (np.ndarray, tf.Tensor)):
                v: BackingTensor = _encode(initial_value, apply_scaling)
                v0, v1 = _share(v)

            elif isinstance(initial_value, PondPublicTensor):
                v_on_0, _ = initial_value.unwrapped
                with tf.device(self.server_0.device_name):
                    # NOTE[Morten]
                    # we can alternatively avoid transfer of v1 from server0 and server1
                    # by having the crypto producer (pre-)generate sharings of zero
                    v0, v1 = _share(v_on_0)

            elif isinstance(initial_value, PondPrivateTensor):
                v0, v1 = initial_value.unwrapped

            else:
                raise TypeError(
                    "Don't know how to turn {} into private variable".format(type(initial_value)))

            with tf.device(self.server_0.device_name):
                x0 = BackingVariable.from_int100(v0)

            with tf.device(self.server_1.device_name):
                x1 = BackingVariable.from_int100(v1)

        x = PondPrivateVariable(self, x0, x1)
        _initializers.append(x.initializer)
        return x

    def define_public_input(
        self,
        provider: InputProvider,
        apply_scaling: bool=True,
        name: str=None
    ) -> Union['PondPublicTensor', List['PondPublicTensor']]:

        def helper(v: tf.Tensor):
            assert v.shape.is_fully_defined(), "Shape of input '{}' on '{}' is not fully defined".format(name if name else '', provider.player.name)
            v: BackingTensor = _encode(v, apply_scaling)
            return PondPublicTensor(self, v, v)

        with tf.name_scope('public-input{}'.format('-' + name if name else '')):

            with tf.device(provider.player.device_name):

                inputs = provider.provide_input()

                if isinstance(inputs, tf.Tensor):
                    # single input -> single output
                    v = inputs
                    return helper(v)

                elif isinstance(inputs, (list, tuple)):
                    # multiple inputs -> multiple outputs
                    return [helper(v) for v in inputs]

                else:
                    raise TypeError("Don't know how to handle inputs of type {}".format(type(inputs)))

    def define_private_input(
        self,
        provider: InputProvider,
        apply_scaling: bool=True,
        name: str=None,
        masked: bool=False
    ) -> Union['PondPrivateTensor', 'PondMaskedTensor', List['PondPrivateTensor'], List['PondMaskedTensor']]:

        def helper(v: tf.Tensor):
            assert v.shape.is_fully_defined(), "Shape of input '{}' on '{}' is not fully defined".format(name if name else '', provider.player.name)

            v = _encode(v, apply_scaling)
            x0, x1 = _share(v)
            x = PondPrivateTensor(self, x0, x1)

            if masked:
                with tf.name_scope('local_mask'):
                    a = BackingTensor.sample_uniform(v.shape)
                    a0, a1 = _share(a)
                    alpha = v - a
                x = PondMaskedTensor(self, x, a, a0, a1, alpha, alpha)

            return x

        with tf.name_scope('private-input{}'.format('-' + name if name else '')):

            with tf.device(provider.player.device_name):

                inputs = provider.provide_input()

                if isinstance(inputs, tf.Tensor):
                    # single input -> single output
                    v = inputs
                    output = helper(v)

                elif isinstance(inputs, (list, tuple)):
                    # multiple inputs -> multiple outputs
                    output = [helper(v) for v in inputs]

                else:
                    raise TypeError("Don't know how to handle inputs of type {}".format(type(inputs)))

        return output

    def define_output(
        self,
        xs: Union['PondPrivateTensor', List['PondPrivateTensor']],
        receiver: OutputReceiver,
        apply_scaling: bool=True,
        name: Optional[str]=None
    ) -> tf.Operation:

        def helper(x: 'PondPrivateTensor'):
            assert isinstance(x, PondPrivateTensor), type(x)
            x0, x1 = x.unwrapped
            v: BackingTensor = _reconstruct(x0, x1)
            v: tf.Tensor = _decode(v, apply_scaling)
            return v

        with tf.name_scope('output{}'.format('-' + name if name else '')):

            with tf.device(receiver.player.device_name):

                if isinstance(xs, PondPrivateTensor):
                    # single input -> single output
                    x = xs
                    op = receiver.receive_output(helper(x))

                elif isinstance(xs, (list, tuple)):
                    op = receiver.receive_output([helper(x) for x in xs])

                else:
                    raise TypeError("Don't know how to handle inputs of type {}".format(type(xs)))

                # wrap in tf.group to prevent sending back any tensors (which might hence be leaked)
                op = tf.group(op)

        return op

    @property
    def initializer(self) -> tf.Operation:
        return tf.group(*_initializers)

    def clear_initializers(self) -> None:
        del _initializers[:]

    def assign(self, variable, value):
        assert isinstance(variable, PondPrivateVariable), type(variable)
        assert isinstance(value, PondPrivateTensor), type(value)

        node_key = ('assign', variable, value)
        op = _nodes.get(node_key, None)

        if op is not None:
            return op

        var0, var1 = variable.variable0, variable.variable1
        val0, val1 = value.share0, value.share1

        with tf.name_scope('assign'):

            with tf.device(self.server_0.device_name):
                op0 = var0.assign_from_int100(val0)

            with tf.device(self.server_1.device_name):
                op1 = var1.assign_from_int100(val1)

        op = tf.group(op0, op1)
        _nodes[node_key] = op

        return op

    def add(self, x, y):

        node_key = ('add', x, y)
        z = _nodes.get(node_key, None)

        if z is not None:
            return z

        x = _lift(self, x)
        y = _lift(self, y)

        dispatch = {
            (PondPublicTensor, PondPublicTensor): _add_public_public,
            (PondPublicTensor, PondPrivateTensor): _add_public_private,
            (PondPublicTensor, PondMaskedTensor): _add_public_masked,
            (PondPrivateTensor, PondPublicTensor): _add_private_public,
            (PondPrivateTensor, PondPrivateTensor): _add_private_private,
            (PondPrivateTensor, PondMaskedTensor): _add_private_masked,
            (PondMaskedTensor, PondPublicTensor): _add_masked_public,
            (PondMaskedTensor, PondPrivateTensor): _add_masked_private,
            (PondMaskedTensor, PondMaskedTensor): _add_masked_masked
        }
        func = dispatch.get((_type(x), _type(y)), None)
        if func is None:
            raise TypeError("Don't know how to add {} and {}".format(type(x), type(y)))

        z = func(self, x, y)
        _nodes[node_key] = z

        return z

    def sum(self, x, axis, keepdims):
        node_key = ('sum', x)
        z = _nodes.get(node_key, None)

        if z is not None:
            return z

        x = _lift(self, x)

        dispatch = {
            PondPublicTensor: _sum_public,
            PondPrivateTensor: _sum_private,
            PondMaskedTensor: _sum_masked
        }
        func = dispatch.get(_type(x), None)
        if func is None:
            raise TypeError("Don't know how to sum {}".format(type(x)))

        z = func(self, x, axis, keepdims)
        _nodes[node_key] = z

        return z

    def sub(self, x, y):

        node_key = ('sub', x, y)
        z = _nodes.get(node_key, None)

        if z is not None:
            return z

        x = _lift(self, x)
        y = _lift(self, y)

        dispatch = {
            (PondPublicTensor, PondPublicTensor): _sub_public_public,
            (PondPublicTensor, PondPrivateTensor): _sub_public_private,
            (PondPublicTensor, PondMaskedTensor): _sub_public_masked,
            (PondPrivateTensor, PondPublicTensor): _sub_private_public,
            (PondPrivateTensor, PondPrivateTensor): _sub_private_private,
            (PondPrivateTensor, PondMaskedTensor): _sub_private_masked,
            (PondMaskedTensor, PondPublicTensor): _sub_masked_public,
            (PondMaskedTensor, PondPrivateTensor): _sub_masked_private,
            (PondMaskedTensor, PondMaskedTensor): _sub_masked_masked
        }
        func = dispatch.get((_type(x), _type(y)), None)
        if func is None:
            raise TypeError("Don't know how to sub {} and {}".format(type(x), type(y)))

        z = func(self, x, y)
        _nodes[node_key] = z

        return z

    def mask(self, x):

        if isinstance(x, (list, tuple)):
            # apply recursively
            return [self.mask(xi) for xi in x]

        node_key = ('mask', x)
        x_masked = _nodes.get(node_key, None)

        if x_masked is not None:
            return x_masked

        if isinstance(x, PondPrivateTensor):
            x_masked = _mask_private(self, x)

        else:
            raise TypeError("Don't know how to mask {}".format(type(x)))

        _nodes[node_key] = x_masked
        return x_masked

    def mul(self, x, y):

        node_key = ('mul', x, y)
        z = _nodes.get(node_key, None)

        if z is not None:
            return z

        x = _lift(self, x)
        y = _lift(self, y)

        dispatch = {
            (PondPublicTensor, PondPublicTensor): _mul_public_public,
            (PondPublicTensor, PondPrivateTensor): _mul_public_private,
            (PondPublicTensor, PondMaskedTensor): _mul_public_masked,
            (PondPrivateTensor, PondPublicTensor): _mul_private_public,
            (PondPrivateTensor, PondPrivateTensor): _mul_private_private,
            (PondPrivateTensor, PondMaskedTensor): _mul_private_masked,
            (PondMaskedTensor, PondPublicTensor): _mul_masked_public,
            (PondMaskedTensor, PondPrivateTensor): _mul_masked_private,
            (PondMaskedTensor, PondMaskedTensor): _mul_masked_masked
        }
        func = dispatch.get((_type(x), _type(y)), None)
        if func is None:
            raise TypeError("Don't know how to mul {} and {}".format(type(x), type(y)))

        z = func(self, x, y)
        _nodes[node_key] = z

        return z

    def square(self, x):

        node_key = ('square', x)
        y = _nodes.get(node_key, None)

        if y is not None:
            return y

        if isinstance(x, PondPublicTensor):
            y = _square_public(self, x)

        elif isinstance(x, PondPrivateTensor):
            y = _square_private(self, x)

        elif isinstance(x, PondMaskedTensor):
            y = _square_masked(self, x)

        else:
            raise TypeError("Don't know how to square {}".format(type(x)))

        _nodes[node_key] = y

        return y

    def dot(self, x, y):

        node_key = ('dot', x, y)
        z = _nodes.get(node_key, None)

        if z is not None:
            return z

        dispatch = {
            (PondPublicTensor, PondPublicTensor): _dot_public_public,
            (PondPublicTensor, PondPrivateTensor): _dot_public_private,
            (PondPublicTensor, PondMaskedTensor): _dot_public_masked,
            (PondPrivateTensor, PondPublicTensor): _dot_private_public,
            (PondPrivateTensor, PondPrivateTensor): _dot_private_private,
            (PondPrivateTensor, PondMaskedTensor): _dot_private_masked,
            (PondMaskedTensor, PondPublicTensor): _dot_masked_public,
            (PondMaskedTensor, PondPrivateTensor): _dot_masked_private,
            (PondMaskedTensor, PondMaskedTensor): _dot_masked_masked
        }
        func = dispatch.get((_type(x), _type(y)), None)
        if func is None:
            raise TypeError("Don't know how to mul {} and {}".format(type(x), type(y)))

        z = func(self, x, y)
        _nodes[node_key] = z

        return z

    def truncate(self, x: 'PondTensor'):

        node_key = ('truncate', x)
        y = _nodes.get(node_key, None)

        if y is not None:
            return y

        if isinstance(x, PondPublicTensor):
            y = _truncate_public(self, x)

        elif isinstance(x, PondPrivateTensor):
            y = _truncate_private(self, x)

        elif isinstance(x, PondMaskedTensor):
            y = _truncate_masked(self, x)

        else:
            raise TypeError("Don't know how to truncate {}".format(type(x)))

        _nodes[node_key] = y

        return y

    def transpose(self, x: 'PondTensor', perm=None):

        node_key = ('transpose', x)
        x_t = _nodes.get(node_key, None)

        if x_t is not None:
            return x_t

        if isinstance(x, PondPublicTensor):
            x_t = _transpose_public(self, x, perm=perm)

        elif isinstance(x, PondPrivateTensor):
            x_t = _transpose_private(self, x, perm=perm)

        elif isinstance(x, PondMaskedTensor):
            x_t = _transpose_masked(self, x, perm=perm)
            _nodes[('transpose', x.unmasked)] = x_t.unmasked

        else:
            raise TypeError("Don't know how to transpose {}".format(type(x)))

        _nodes[node_key] = x_t

        return x_t

    def reshape(self, x: 'PondTensor', shape: List[int]):

        node_key = ('reshape', x)
        x_reshaped = _nodes.get(node_key, None)

        if x_reshaped is not None:
            return x_reshaped

        if isinstance(x, PondPublicTensor):
            x_reshaped = _reshape_public(self, x, shape)

        elif isinstance(x, PondPrivateTensor):
            x_reshaped = _reshape_private(self, x, shape)

        elif isinstance(x, PondMaskedTensor):
            x_reshaped = _reshape_masked(self, x, shape)
            _nodes[('reshape', x.unmasked)] = x_reshaped.unmasked

        else:
            raise TypeError("Don't know how to reshape {}".format(type(x)))

        _nodes[node_key] = x_reshaped

        return x_reshaped

    def expand_dims(self, x: 'PondTensor', axis=None):

        node_key = ('expand', x)
        x_e = _nodes.get(node_key, None)

        if x_e is not None:
            return x_e

        if isinstance(x, PondPublicTensor):
            x_e = _expand_dims_public(self, x, axis=axis)

        elif isinstance(x, PondPrivateTensor):
            x_e = _expand_dims_private(self, x, axis=axis)

        elif isinstance(x, PondMaskedTensor):
            x_e = _expand_dims_masked(self, x, axis=axis)
            _nodes[('expand', x.unmasked)] = x_e.unmasked

        else:
            raise TypeError("Don't know how to expand dims {}".format(type(x)))

        _nodes[node_key] = x_e

        return x_e

    def squeeze(self, x: 'PondTensor', axis: List[int]):

        node_key = ('squeeze', x)
        x_squeezed = _nodes.get(node_key, None)

        if x_squeezed is not None:
            return x_squeezed

        if isinstance(x, PondPublicTensor):
            x_squeezed = _squeeze_public(self, x, axis)

        elif isinstance(x, PondPrivateTensor):
            x_squeezed = _squeeze_private(self, x, axis)

        elif isinstance(x, PondMaskedTensor):
            x_squeezed = _squeeze_masked(self, x, axis)
            _nodes[('sqeeze', x.unmasked)] = x_squeezed.unmasked

        else:
            raise TypeError("Don't know how to squeeze {}".format(type(x)))

        _nodes[node_key] = x_squeezed

        return x_squeezed

    def strided_slice(self, x: 'PondTensor', *args: Any, **kwargs: Any):
        """ See https://www.tensorflow.org/api_docs/python/tf/strided_slice for documentation on the arguments """

        node_key = ('strided_slice', x)

        x_sliced = _nodes.get(node_key, None)

        if x_sliced is not None:
            return x_sliced

        if isinstance(x, PondPublicTensor):
            x_sliced = _strided_slice_public(self, x, args, kwargs)
        elif isinstance(x, PondPrivateTensor):
            x_sliced = _strided_slice_private(self, x, args, kwargs)
        elif isinstance(x, PondMaskedTensor):
            x_sliced = _strided_slice_masked(self, x, args, kwargs)
            _nodes[('strided_slice', x.unmasked)] = x_sliced.unmasked
        else:
            raise TypeError("Don't know how to do a strided slice {}".format(type(x)))

        _nodes[node_key] = x_sliced

        return x_sliced

    def stack(self, xs: List['PondTensor'], axis: int = 0):

        node_key = ('stack', tuple(xs))
        xs_stack = _nodes.get(node_key, None)

        if xs_stack is not None:
            return xs_stack

        if all([isinstance(x, PondPublicTensor) for x in xs]):
            xs_stack = _stack_public(self, xs, axis=axis)

        elif all([isinstance(x, PondPrivateTensor) for x in xs]):
            xs_stack = _stack_private(self, xs, axis=axis)

        elif all([isinstance(x, PondMaskedTensor) for x in xs]):
            xs_stack = _stack_masked(self, xs, axis=axis)

            unmasked = [x.unmasked for x in xs_stack]
            _nodes[('stack', unmasked)] = unmasked

        else:
            raise TypeError("Don't know how to do a stack {}".format(type(xs)))

        _nodes[node_key] = xs_stack

        return xs_stack

    def sigmoid(self, x: 'PondTensor'):
        assert isinstance(x, PondTensor), type(x)

        node_key = ('sigmoid', x)
        z = _nodes.get(node_key, None)

        if z is not None:
            return z

        w0 = 0.5
        w1 = 0.2159198015
        w3 = -0.0082176259
        w5 = 0.0001825597
        w7 = -0.0000018848
        w9 = 0.0000000072

        with tf.name_scope('sigmoid'):

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

        _nodes[node_key] = z

        return z

    def relu(self, x: 'PondTensor'):
        assert isinstance(x, PondTensor), type(x)

        node_key = ('relu', x)
        z = _nodes.get(node_key, None)

        if z is not None:
            return z

        w0 = 0.44015372000819103
        w1 = 0.500000000
        w2 = 0.11217537671414643
        w4 = -0.0013660836712429923
        w6 = 9.009136367360004e-06
        w8 = -2.1097433984e-08

        with tf.name_scope('relu'):

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

        _nodes[node_key] = z

        return z

    def reveal(self, x):

        node_key = ('reveal', x)
        z = _nodes.get(node_key, None)

        if z is not None:
            return z

        dispatch = {
            PondPrivateTensor: _reveal_private,
            PondMaskedTensor: _reveal_masked
        }
        func = dispatch.get(_type(x), None)
        if func is None:
            raise TypeError("Don't know how to reveal {}".format(_type(x)))

        z = func(self, x)
        _nodes[node_key] = z

        return z

    def cache(self, x):

        if isinstance(x, (list, tuple)):
            # apply recursively
            return [self.cache(xi) for xi in x]

        node_key = ('cache', x)
        cached = _nodes.get(node_key, None)

        if cached is not None:
            return cached

        dispatch = {
            PondPublicTensor: _cache_public,
            PondPrivateTensor: _cache_private,
            PondMaskedTensor: _cache_masked
        }
        func = dispatch.get(_type(x), None)
        if func is None:
            raise TypeError("Don't know how to cache {}".format(type(x)))

        cached = func(self, x)
        _nodes[node_key] = cached

        return cached

    def conv2d(self, x, w, strides, padding):

        node_key = ('conv2d', x, w, strides, padding)
        z = _nodes.get(node_key, None)

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
            (PondMaskedTensor, PondMaskedTensor): _conv2d_masked_masked
        }

        func = dispatch.get((_type(x), _type(w)), None)
        if func is None:
            raise TypeError("Don't know how to conv2d {} and {}".format(type(x), type(w)))

        z = func(self, x, w, strides, padding)
        _nodes[node_key] = z

        return z

    def avgpool2d(self, x, pool_size, strides, padding):
        node_key = ('avgpool2d', x, tuple(pool_size), tuple(strides), padding)
        z = _nodes.get(node_key, None)

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
        _nodes[node_key] = z

        return z


#
# Classes representing the base values in the Pond protocol.
#


class PondTensor(object):
    """
    This class functions mostly as a convenient way of exposing operations
    directly on the various tensor objects, ie allowing one to write x + y
    instead of prot.add(x, y). Since this functionality is shared among all
    tensors we put it in this superclass.

    This class should never be instantiated on its own.
    TODO[Morten] make it abstract
    """

    def __init__(self, prot):
        self.prot = prot

    def add(self, other):
        return self.prot.add(self, other)

    def __add__(self, other):
        return self.prot.add(self, other)

    def sum(self, axis, keepdims=False):
        return self.prot.sum(self, axis, keepdims)

    def sub(self, other):
        return self.prot.sub(self, other)

    def __sub__(self, other):
        return self.prot.sub(self, other)

    def mul(self, other):
        return self.prot.mul(self, other)

    def __mul__(self, other):
        return self.prot.mul(self, other)

    def square(self):
        return self.prot.square(self)

    def dot(self, other):
        return self.prot.dot(self, other)

    def tranpose(self):
        return self.prot.transpose(self)

    def truncate(self):
        return self.prot.truncate(self)

    def expand_dims(self):
        return self.prot.expand_dims(self)


class PondPublicTensor(PondTensor):
    """
    This class represents a public tensor, known by at least the two servers
    but potentially known by more. Although there is only a single value we
    replicate it on both servers to avoid sending it from one to the other
    in the operations where it's needed by both (eg multiplication).
    """

    def __init__(self, prot, value_on_0: BackingTensor, value_on_1: BackingTensor) -> None:
        assert isinstance(value_on_0, BackingTensor), type(value_on_0)
        assert isinstance(value_on_1, BackingTensor), type(value_on_1)
        assert value_on_0.shape == value_on_1.shape

        super(PondPublicTensor, self).__init__(prot)
        self.value_on_0 = value_on_0
        self.value_on_1 = value_on_1
        self.encoded = True  # TODO[Morten] take as parameter

    def __repr__(self):
        return 'PondPublicTensor(shape={})'.format(self.shape)

    @property
    def shape(self):
        return self.value_on_0.shape

    @property
    def unwrapped(self):
        return (self.value_on_0, self.value_on_1)

    def eval(self, sess, feed_dict={}, tag=None):
        value: BackingTensor = self.value_on_0.eval(sess, feed_dict=feed_dict, tag=tag)
        return _decode(value, self.encoded)


class PondPrivateTensor(PondTensor):
    """
    This class represents a private value that may be unknown to everyone.
    """

    def __init__(self, prot, share0: BackingTensor, share1: BackingTensor) -> None:
        assert isinstance(share0, BackingTensor), type(share0)
        assert isinstance(share1, BackingTensor), type(share1)
        assert share0.shape == share1.shape

        super(PondPrivateTensor, self).__init__(prot)
        self.share0 = share0
        self.share1 = share1

    def __repr__(self):
        return 'PondPrivateTensor(shape={})'.format(self.shape)

    @property
    def shape(self):
        return self.share0.shape

    @property
    def unwrapped(self):
        return (self.share0, self.share1)

    def reveal(self):
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

    def __init__(self, prot, unmasked, a, a0, a1, alpha_on_0, alpha_on_1):
        assert isinstance(unmasked, PondPrivateTensor)

        super(PondMaskedTensor, self).__init__(prot)
        self.unmasked = unmasked
        self.a = a
        self.a0 = a0
        self.a1 = a1
        self.alpha_on_0 = alpha_on_0
        self.alpha_on_1 = alpha_on_1

    def __repr__(self):
        return 'PondMaskedTensor(shape={})'.format(self.shape)

    @property
    def shape(self):
        return self.a.shape

    @property
    def unwrapped(self):
        return (self.a, self.a0, self.a1, self.alpha_on_0, self.alpha_on_1)


#
# Extentions of the base Pond classes that record extra information
# relevant to how TensorFlow works.
#


class PondConstant(PondPublicTensor):
    """
    This class essentially represents a public value, however it additionally
    records the fact that the underlying value was declared as a constant.
    """

    def __init__(self, prot, constant_on_0, constant_on_1):
        assert type(constant_on_0) is BackingConstant, type(constant_on_0)
        assert type(constant_on_1) is BackingConstant, type(constant_on_1)
        assert constant_on_0.shape == constant_on_1.shape

        super(PondConstant, self).__init__(prot, constant_on_0, constant_on_1)
        self.constant_on_0 = constant_on_0
        self.constant_on_1 = constant_on_1

    def __repr__(self):
        return 'PondConstant(shape={})'.format(self.shape)


class PondPublicPlaceholder(PondPublicTensor):
    """
    This class essentially represents a public value, however it additionally
    records the fact that the backing tensor was declared as a placeholder in
    order to allow treating it as a placeholder itself.
    """

    def __init__(self, prot, placeholder_on_0, placeholder_on_1):
        assert type(placeholder_on_0) is BackingPlaceholder, type(placeholder_on_0)
        assert type(placeholder_on_1) is BackingPlaceholder, type(placeholder_on_1)
        assert placeholder_on_0.shape == placeholder_on_1.shape

        super(PondPublicPlaceholder, self).__init__(prot, placeholder_on_0, placeholder_on_1)
        self.placeholder_on_0 = placeholder_on_0
        self.placeholder_on_1 = placeholder_on_1

    def __repr__(self):
        return 'PondPublicPlaceholder(shape={})'.format(self.shape)


class PondPrivatePlaceholder(PondPrivateTensor):
    """
    This class essentially represents a private value, however it additionally
    records the fact that the backing tensor was declared as a placeholder in
    order to allow treating it as a placeholder itself.
    """

    def __init__(self, prot, placeholder, tensor0, tensor1):
        assert type(placeholder) is BackingPlaceholder, type(placeholder)
        assert type(tensor0) is BackingTensor, type(tensor0)
        assert type(tensor1) is BackingTensor, type(tensor1)
        assert tensor0.shape == tensor1.shape

        super(PondPrivatePlaceholder, self).__init__(prot, tensor0, tensor1)
        self.placeholders = placeholder.backing
        self.tensor0 = tensor0
        self.tensor1 = tensor1

    def __repr__(self):
        return 'PondPrivatePlaceholder(shape={})'.format(self.shape)

    def feed_from_native(self, value, apply_scaling=True):
        assert type(value) in [np.ndarray], type(value)

        v = _encode(value, apply_scaling)
        return {
            p: v for p, v in zip(self.placeholders, v.backing)
        }


class PondPublicVariable(PondPublicTensor):
    """
    This class essentially represents a public value, however it additionally
    records the fact that the backing tensor was declared as a variable in
    order to allow treating it as a variable itself.
    """

    def __init__(self, prot, variable_on_0, variable_on_1):
        assert type(variable_on_0) is BackingVariable, type(variable_on_0)
        assert type(variable_on_1) is BackingVariable, type(variable_on_1)
        assert variable_on_0.shape == variable_on_1.shape

        super(PondPublicVariable, self).__init__(prot, variable_on_0, variable_on_1)
        self.variable_on_0 = variable_on_0
        self.variable_on_1 = variable_on_1
        self.initializer = tf.group(*[var.initializer for var in [variable_on_0, variable_on_1]])

    def __repr__(self):
        return 'PondPublicVariable(shape={})'.format(self.shape)


class PondPrivateVariable(PondPrivateTensor):
    """
    This class essentially represents a private value, however it additionally
    records the fact that the backing tensor was declared as a variable in
    order to allow treating it as a variable itself.
    """

    def __init__(self, prot, variable0, variable1):
        assert type(variable0) is BackingVariable, type(variable0)
        assert type(variable1) is BackingVariable, type(variable1)
        assert variable0.shape == variable1.shape

        super(PondPrivateVariable, self).__init__(prot, variable0, variable1)
        self.variable0 = variable0
        self.variable1 = variable1
        self.initializer = tf.group(*[var.initializer for var in [variable0, variable1]])

    def __repr__(self):
        return 'PondPrivateVariable(shape={})'.format(self.shape)


class PondCachedPublicTensor(PondPrivateTensor):

    def __init__(self, prot, x_on_0, x_on_1, updator):
        assert isinstance(x_on_0, BackingTensor), type(x_on_0)
        assert isinstance(x_on_1, BackingTensor), type(x_on_1)
        assert isinstance(updator, tf.Operation), type(updator)

        super(PondCachedPublicTensor, self).__init__(prot, x_on_0, x_on_1)
        self.updator = updator

    def __repr__(self):
        return 'PondCachedPublicTensor(shape={})'.format(self.shape)


class PondCachedPrivateTensor(PondPrivateTensor):

    def __init__(self, prot, x0, x1, updator):
        assert isinstance(x0, BackingTensor), type(x0)
        assert isinstance(x1, BackingTensor), type(x1)
        assert isinstance(updator, tf.Operation), type(updator)

        super(PondCachedPrivateTensor, self).__init__(prot, x0, x1)
        self.updator = updator

    def __repr__(self):
        return 'PondCachedPrivateTensor(shape={})'.format(self.shape)


class PondCachedMaskedTensor(PondMaskedTensor):

    def __init__(self, prot, unmasked, a, a0, a1, alpha_on_0, alpha_on_1, updator):
        assert isinstance(unmasked, PondPrivateTensor), type(unmasked)
        assert isinstance(a, BackingTensor), type(a)
        assert isinstance(a0, BackingTensor), type(a0)
        assert isinstance(a1, BackingTensor), type(a1)
        assert isinstance(alpha_on_0, BackingTensor), type(alpha_on_0)
        assert isinstance(alpha_on_1, BackingTensor), type(alpha_on_1)
        assert isinstance(updator, tf.Operation), type(updator)

        super(PondCachedMaskedTensor, self).__init__(prot, unmasked, a, a0, a1, alpha_on_0, alpha_on_1)
        self.updator = updator

    def __repr__(self):
        return 'PondCachedMaskedTensor(shape={})'.format(self.shape)


def _encode(rationals, apply_scaling) -> BackingTensor:
    """ Encode tensor of rational numbers into tensor of ring elements """

    with tf.name_scope('encode'):

        scaling_factor = 2 ** BITPRECISION_FRACTIONAL if apply_scaling else 1

        if isinstance(rationals, np.ndarray):
            encoded = (rationals * scaling_factor).astype(int).astype(object)

        elif isinstance(rationals, tf.Tensor):
            encoded = tf.cast(rationals * scaling_factor, tf.int32)

        else:
            raise TypeError("Don't know how to encode {}".format(type(rationals)))

        return BackingTensor.from_native(encoded)


def _decode(elements: BackingTensor, apply_scaling: bool):
    """ Decode tensor of ring elements into tensor of rational numbers """

    with tf.name_scope('decode'):

        scaling_factor = 2 ** BITPRECISION_FRACTIONAL if apply_scaling else 1

        # NOTE we assume that x + BOUND fits within int32, ie that (BOUND - 1) + BOUND <= 2**31 - 1
        return ((elements + BOUND).to_int32() - BOUND) / scaling_factor


def _share(secret: BackingTensor) -> Tuple[BackingTensor, BackingTensor]:

    with tf.name_scope('share'):
        share0 = BackingTensor.sample_uniform(secret.shape)
        share1 = secret - share0
        return share0, share1


def _reconstruct(share0: BackingTensor, share1: BackingTensor):

    with tf.name_scope('reconstruct'):
        return share0 + share1


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


def _lift(prot, x):
    """
    Convenience method for working with constants in programs: mixing any of the
    Pond objects together with eg ints and floats will automatically lift the
    latter into Pond objects.
    """

    if isinstance(x, (PondPublicTensor, PondPrivateTensor, PondMaskedTensor)):
        # don't do anthing to these
        return x

    if type(x) is int:
        return prot.define_constant(np.array([x]))

    if type(x) is float:
        return prot.define_constant(np.array([x]))

    raise TypeError("Don't know how to lift {}".format(type(x)))


#
# cache
#


def _cache_wrap_helper(sources):
    variables = [
        BackingVariable.from_native(tf.zeros(shape=source.shape, dtype=BackingTensor.int_type))
        for source in sources
    ]
    updator = tf.group(*[
        var.assign_from_int100(val)
        for var, val in zip(variables, sources)
    ])
    return variables, updator


def _cache_public(prot, x):
    assert isinstance(x, PondPublicTensor), type(x)

    x_on_0, x_on_1 = x.unwrapped

    with tf.name_scope('cache'):

        with tf.device(prot.server_0.device_name):
            [x_on_0_cached], updator0 = _cache_wrap_helper([x_on_0])

        with tf.device(prot.server_1.device_name):
            [x_on_1_cached], updator1 = _cache_wrap_helper([x_on_1])

        updator = tf.group(updator0, updator1)

    _global_cache_updators.append(updator)
    return PondCachedPublicTensor(
        prot,
        x_on_0_cached,
        x_on_1_cached,
        updator
    )


def _cache_private(prot, x):
    assert isinstance(x, PondPrivateTensor), type(x)

    x0, x1 = x.unwrapped

    with tf.name_scope('cache'):

        with tf.device(prot.server_0.device_name):
            [x0_cached], updator0 = _cache_wrap_helper([x0])

        with tf.device(prot.server_1.device_name):
            [x1_cached], updator1 = _cache_wrap_helper([x1])

        updator = tf.group(updator0, updator1)

    _global_cache_updators.append(updator)
    return PondCachedPrivateTensor(
        prot,
        x0_cached,
        x1_cached,
        updator
    )


def _cache_masked(prot, x):
    assert isinstance(x, PondMaskedTensor), type(x)

    unmasked = x.unmasked
    a, a0, a1, alpha_on_0, alpha_on_1 = x.unwrapped

    with tf.name_scope('cache'):

        with tf.device(prot.crypto_producer.device_name):
            [a_cached], updator_cp = _cache_wrap_helper([a])

        with tf.device(prot.server_0.device_name):
            [a0_cached, alpha_on_0_cached], updator0 = _cache_wrap_helper([a0, alpha_on_0])

        with tf.device(prot.server_1.device_name):
            [a1_cached, alpha_on_1_cached], updator1 = _cache_wrap_helper([a1, alpha_on_1])

        updator = tf.group(updator_cp, updator0, updator1)
        unmasked_cached = prot.cache(unmasked)

    _global_cache_updators.append(updator)
    return PondCachedMaskedTensor(
        prot,
        unmasked_cached,
        a_cached,
        a0_cached,
        a1_cached,
        alpha_on_0_cached,
        alpha_on_1_cached,
        updator
    )


#
# truncate
#

# precomputation
K_inv = BackingTensor.from_native(np.array([inverse(K, M)]))
M_wrapped = BackingTensor.from_native(np.array([M]))


def _raw_truncate(x):
    y = x - (x % K)
    return y * K_inv


def _truncate_public(prot, x):
    assert isinstance(x, PondPublicTensor)

    x_on_0, x_on_1 = x.unwrapped

    with tf.name_scope('truncate'):

        with tf.device(prot.server_0.device_name):
            y_on_0 = _raw_truncate(x_on_0)

        with tf.device(prot.server_1.device_name):
            y_on_1 = _raw_truncate(x_on_1)

    return PondPublicTensor(prot, y_on_0, y_on_1)


def _truncate_private(prot, x):
    assert isinstance(x, PondPrivateTensor)

    x0, x1 = x.unwrapped

    with tf.name_scope('truncate'):

        with tf.device(prot.server_0.device_name):
            y0 = _raw_truncate(x0)

        with tf.device(prot.server_1.device_name):
            y1 = M_wrapped - _raw_truncate(M_wrapped - x1)

    return PondPrivateTensor(prot, y0, y1)


def _truncate_masked(prot, x):
    assert isinstance(x, PondMaskedTensor)
    return prot.truncate(x.unmasked)


#
# reveal helpers
#


def _reveal_private(prot, x):
    assert isinstance(x, PondPrivateTensor), type(x)

    with tf.name_scope('reveal'):

        x0, x1 = x.unwrapped

        with tf.device(prot.server_0.device_name):
            z_on_0 = x0 + x1

        with tf.device(prot.server_1.device_name):
            z_on_1 = x0 + x1

    z = PondPublicTensor(prot, z_on_0, z_on_1)
    return z


def _reveal_masked(prot, x):
    assert isinstance(x, PondMaskedTensor), type(x)
    return prot.reveal(x.unmasked)


#
# add helpers
#


def _add_public_public(prot, x, y):
    assert isinstance(x, PondPublicTensor), type(x)
    assert isinstance(y, PondPublicTensor), type(y)

    x_on_0, x_on_1 = x.unwrapped
    y_on_0, y_on_1 = y.unwrapped

    with tf.name_scope('add'):

        with tf.device(prot.server_0.device_name):
            z_on_0 = x_on_0 + y_on_0

        with tf.device(prot.server_1.device_name):
            z_on_1 = x_on_1 + y_on_1

    z = PondPublicTensor(prot, z_on_0, z_on_1)
    return z


def _add_public_private(prot, x, y):
    assert isinstance(x, PondPublicTensor), type(x)
    assert isinstance(y, PondPrivateTensor), type(y)

    x_on_0, _ = x.unwrapped
    y0, y1 = y.unwrapped

    with tf.name_scope('add'):

        with tf.device(prot.server_0.device_name):
            z0 = x_on_0 + y0

        with tf.device(prot.server_1.device_name):
            z1 = y1

    z = PondPrivateTensor(prot, z0, z1)
    return z


def _add_public_masked(prot, x, y):
    assert isinstance(x, PondPublicTensor), type(x)
    assert isinstance(y, PondMaskedTensor), type(y)
    return prot.add(x, y.unmasked)


def _add_private_public(prot, x, y):
    assert isinstance(x, PondPrivateTensor), type(x)
    assert isinstance(y, PondPublicTensor), type(y)

    x0, x1 = x.unwrapped
    y_on_0, _ = y.unwrapped

    with tf.name_scope('add'):

        with tf.device(prot.server_0.device_name):
            z0 = x0 + y_on_0

        with tf.device(prot.server_1.device_name):
            z1 = x1

    z = PondPrivateTensor(prot, z0, z1)
    return z


def _add_private_private(prot, x, y):
    assert isinstance(x, PondPrivateTensor), type(x)
    assert isinstance(y, PondPrivateTensor), type(y)

    x0, x1 = x.unwrapped
    y0, y1 = y.unwrapped

    with tf.name_scope('add'):

        with tf.device(prot.server_0.device_name):
            z0 = x0 + y0

        with tf.device(prot.server_1.device_name):
            z1 = x1 + y1

    z = PondPrivateTensor(prot, z0, z1)
    return z


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
# sum helpers
#


def _sum_core(prot: Pond,
              x: PondTensor,
              axis: int,
              keepdims: Optional[bool]) -> Tuple[BackingTensor, BackingTensor]:
    x_on_0, x_on_1 = x.unwrapped

    with tf.name_scope('sum'):

        with tf.device(prot.server_0.device_name):
            y_on_0 = x_on_0.sum(axis, keepdims)

        with tf.device(prot.server_1.device_name):
            y_on_1 = x_on_1.sum(axis, keepdims)

    return y_on_0, y_on_1


def _sum_public(prot: Pond,
                x: PondPublicTensor,
                axis: int,
                keepdims: Optional[bool]) -> PondPublicTensor:
    y_on_0, y_on_1 = _sum_core(prot, x, axis, keepdims)
    return PondPublicTensor(prot, y_on_0, y_on_1)


def _sum_private(prot: Pond,
                 x: PondPrivateTensor,
                 axis: int,
                 keepdims: Optional[bool]) -> PondPrivateTensor:
    y_on_0, y_on_1 = _sum_core(prot, x, axis, keepdims)
    return PondPrivateTensor(prot, y_on_0, y_on_1)


def _sum_masked(prot: Pond,
                x: PondMaskedTensor,
                axis: int,
                keepdims: Optional[bool]) -> PondPrivateTensor:
    # y_on_0, y_on_1 = _sum_core(prot, x.unmasked, axis, keepdims)
    # return PondPrivateTensor(prot, y_on_0, y_on_1)
    raise NotImplementedError

#
# sub helpers
#


def _sub_public_public(prot, x, y):
    assert isinstance(x, PondPublicTensor), type(x)
    assert isinstance(y, PondPublicTensor), type(y)

    x_on_0, x_on_1 = x.unwrapped
    y_on_0, y_on_1 = y.unwrapped

    with tf.name_scope('sub'):

        with tf.device(prot.server_0.device_name):
            z_on_0 = x_on_0 - y_on_0

        with tf.device(prot.server_1.device_name):
            z_on_1 = x_on_1 - y_on_1

    z = PondPublicTensor(prot, z_on_0, z_on_1)
    return z


def _sub_public_private(prot, x, y):
    assert isinstance(x, PondPublicTensor), type(x)
    assert isinstance(y, PondPrivateTensor), type(y)

    x_on_0, _ = x.unwrapped
    y0, y1 = y.unwrapped

    with tf.name_scope('sub'):

        with tf.device(prot.server_0.device_name):
            z0 = x_on_0 - y0

        with tf.device(prot.server_1.device_name):
            z1 = M - y1

    z = PondPrivateTensor(prot, z0, z1)
    return z


def _sub_public_masked(prot, x, y):
    assert isinstance(x, PondPublicTensor), type(x)
    assert isinstance(y, PondMaskedTensor), type(y)
    return prot.sub(x, y.unmasked)


def _sub_private_public(prot, x, y):
    assert isinstance(x, PondPrivateTensor), type(x)
    assert isinstance(y, PondPublicTensor), type(y)

    x0, x1 = x.unwrapped
    y_on_0, _ = y.unwrapped

    with tf.name_scope('sub'):

        with tf.device(prot.server_0.device_name):
            z0 = x0 - y_on_0

        with tf.device(prot.server_1.device_name):
            z1 = x1

    z = PondPrivateTensor(prot, z0, z1)
    return z


def _sub_private_private(prot, x, y):
    assert isinstance(x, PondPrivateTensor), type(x)
    assert isinstance(y, PondPrivateTensor), type(y)

    x0, x1 = x.unwrapped
    y0, y1 = y.unwrapped

    with tf.name_scope('sub'):

        with tf.device(prot.server_0.device_name):
            z0 = x0 - y0

        with tf.device(prot.server_1.device_name):
            z1 = x1 - y1

    z = PondPrivateTensor(prot, z0, z1)
    return z


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

    with tf.name_scope('mul'):

        with tf.device(prot.server_0.device_name):
            z_on_0 = x_on_0 * y_on_0

        with tf.device(prot.server_1.device_name):
            z_on_1 = x_on_1 * y_on_1

    z = PondPublicTensor(prot, z_on_0, z_on_1)
    z = prot.truncate(z)
    return z


def _mul_public_private(prot, x, y):
    assert isinstance(x, PondPublicTensor), type(x)
    assert isinstance(y, PondPrivateTensor), type(y)

    x_on_0, x_on_1 = x.unwrapped
    y0, y1 = y.unwrapped

    with tf.name_scope('mul'):

        with tf.device(prot.server_0.device_name):
            z0 = x_on_0 * y0

        with tf.device(prot.server_1.device_name):
            z1 = x_on_1 * y1

    z = PondPrivateTensor(prot, z0, z1)
    z = prot.truncate(z)
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

    with tf.name_scope('mul'):

        with tf.device(prot.server_0.device_name):
            z0 = x0 * y_on_0

        with tf.device(prot.server_1.device_name):
            z1 = x1 * y_on_1

    z = PondPrivateTensor(prot, z0, z1)
    z = prot.truncate(z)
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

    with tf.name_scope('mul'):

        with tf.device(prot.crypto_producer.device_name):
            ab = a * b
            ab0, ab1 = _share(ab)

        with tf.device(prot.server_0.device_name):
            alpha = alpha_on_0
            beta = beta_on_0
            z0 = ab0 + (a0 * beta) + (alpha * b0) + (alpha * beta)

        with tf.device(prot.server_1.device_name):
            alpha = alpha_on_1
            beta = beta_on_1
            z1 = ab1 + (a1 * beta) + (alpha * b1)

    z = PondPrivateTensor(prot, z0, z1)
    z = prot.truncate(z)
    return z


#
# square helpers
#


def _square_public(prot, x):
    assert isinstance(x, PondPublicTensor), type(x)

    x_on_0, x_on_1 = x.unwrapped

    with tf.name_scope('square'):

        with tf.device(prot.server_0.device_name):
            y_on_0 = x_on_0 * x_on_0

        with tf.device(prot.server_1.device_name):
            y_on_1 = x_on_1 * x_on_1

    y = PondPublicTensor(prot, y_on_0, y_on_1)
    y = prot.truncate(y)
    return y


def _square_private(prot, x):
    assert isinstance(x, PondPrivateTensor), type(x)
    return prot.square(prot.mask(x))


def _square_masked(prot, x):
    assert isinstance(x, PondMaskedTensor), type(x)

    a, a0, a1, alpha_on_0, alpha_on_1 = x.unwrapped

    with tf.name_scope('square'):

        with tf.device(prot.crypto_producer.device_name):
            aa = a * a
            aa0, aa1 = _share(aa)

        with tf.device(prot.server_0.device_name):
            alpha = alpha_on_0
            # TODO replace with `scale(, 2)` op
            y0 = aa0 + (a0 * alpha) + (alpha * a0) + (alpha * alpha)

        with tf.device(prot.server_1.device_name):
            alpha = alpha_on_1
            # TODO replace with `scale(, 2)` op
            y1 = aa1 + (a1 * alpha) + (alpha * a1)

    y = PondPrivateTensor(prot, y0, y1)
    y = prot.truncate(y)
    return y


#
# dot helpers
#


def _dot_public_public(prot: Pond,
                       x: PondPublicTensor,
                       y: PondPublicTensor) -> PondPublicTensor:

    x_on_0, x_on_1 = x.unwrapped
    y_on_0, y_on_1 = y.unwrapped

    with tf.name_scope('dot'):

        with tf.device(prot.server_0.device_name):
            x = x_on_0
            y = y_on_0
            z_on_0 = x.dot(y)

        with tf.device(prot.server_1.device_name):
            x = x_on_1
            y = y_on_1
            z_on_1 = x.dot(y)

    z = PondPublicTensor(prot, z_on_0, z_on_1)
    z = prot.truncate(z)
    return z


def _dot_public_private(prot, x, y):
    assert isinstance(x, PondPublicTensor), type(x)
    assert isinstance(y, PondPrivateTensor), type(y)

    x_on_0, x_on_1 = x.unwrapped
    y0, y1 = y.unwrapped

    with tf.name_scope('dot'):

        with tf.device(prot.server_0.device_name):
            x = x_on_0
            z0 = x.dot(y0)

        with tf.device(prot.server_1.device_name):
            x = x_on_1
            z1 = x.dot(y1)

    z = PondPrivateTensor(prot, z0, z1)
    z = prot.truncate(z)
    return z


def _dot_public_masked(prot, x, y):
    assert isinstance(x, PondPublicTensor), type(x)
    assert isinstance(y, PondMaskedTensor), type(y)
    return prot.dot(x, y.unmasked)


def _dot_private_public(prot, x, y):
    assert isinstance(x, PondPrivateTensor), type(x)
    assert isinstance(y, PondPublicTensor), type(y)

    x0, x1 = x.unwrapped
    y_on_0, y_on_1 = y.unwrapped

    with tf.name_scope('dot'):

        with tf.device(prot.server_0.device_name):
            y = y_on_0
            z0 = x0.dot(y)

        with tf.device(prot.server_0.device_name):
            y = y_on_1
            z1 = x1.dot(y)

    z = PondPrivateTensor(prot, z0, z1)
    z = prot.truncate(z)
    return z


def _dot_private_private(prot, x, y):
    assert isinstance(x, PondPrivateTensor), type(x)
    assert isinstance(y, PondPrivateTensor), type(y)
    return prot.dot(prot.mask(x), prot.mask(y))


def _dot_private_masked(prot, x, y):
    assert isinstance(x, PondPrivateTensor), type(x)
    assert isinstance(y, PondMaskedTensor), type(y)
    return prot.dot(prot.mask(x), y)


def _dot_masked_public(prot, x, y):
    assert isinstance(x, PondMaskedTensor), type(x)
    assert isinstance(y, PondPublicTensor), type(y)
    return prot.dot(x.unmasked, y)


def _dot_masked_private(prot, x, y):
    assert isinstance(x, PondMaskedTensor), type(x)
    assert isinstance(y, PondPrivateTensor), type(y)
    return prot.dot(x, prot.mask(y))


def _dot_masked_masked(prot, x, y):
    assert isinstance(x, PondMaskedTensor), type(x)
    assert isinstance(y, PondMaskedTensor), type(y)

    a, a0, a1, alpha_on_0, alpha_on_1 = x.unwrapped
    b, b0, b1, beta_on_0, beta_on_1 = y.unwrapped

    with tf.name_scope('dot'):

        with tf.device(prot.crypto_producer.device_name):
            ab = a.dot(b)
            ab0, ab1 = _share(ab)

        with tf.device(prot.server_0.device_name):
            alpha = alpha_on_0
            beta = beta_on_0
            z0 = ab0 + a0.dot(beta) + alpha.dot(b0) + alpha.dot(beta)

        with tf.device(prot.server_1.device_name):
            alpha = alpha_on_1
            beta = beta_on_1
            z1 = ab1 + a1.dot(beta) + alpha.dot(b1)

    z = PondPrivateTensor(prot, z0, z1)
    z = prot.truncate(z)
    return z


#
# Conv helpers
#
# TODO[koen] create operations for all possible combinations


def _conv2d_public_public(prot, x, y, strides, padding):
    raise NotImplementedError()


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

    with tf.name_scope('conv2d'):

        with tf.device(prot.crypto_producer.device_name):
            a_conv2d_b = a.conv2d(b, strides, padding)
            a_conv2d_b0, a_conv2d_b1 = _share(a_conv2d_b)

        with tf.device(prot.server_0.device_name):
            alpha = alpha_on_0
            beta = beta_on_0
            # TODO[koen]: check last term is really conv(alpha,beta) instead of other way around
            z0 = a_conv2d_b0 \
                + a0.conv2d(beta, strides, padding) \
                + alpha.conv2d(b0, strides, padding) \
                + alpha.conv2d(beta, strides, padding)

        with tf.device(prot.server_1.device_name):
            alpha = alpha_on_1
            beta = beta_on_1
            z1 = a_conv2d_b1 \
                + a1.conv2d(beta, strides, padding) \
                + alpha.conv2d(b1, strides, padding)

    z = PondPrivateTensor(prot, z0, z1)
    z = prot.truncate(z)
    return z


#
# average pooling helpers
#


def _avgpool2d_core(prot: Pond,
                    x: PondTensor,
                    pool_size: Tuple[int, int],
                    strides: Tuple[int, int],
                    padding: str) -> Tuple[BackingTensor, BackingTensor, float]:
    x_on_0, x_on_1 = x.unwrapped
    _, _, H, W = x.shape
    scalar = 1 / (pool_size[0] * pool_size[1])
    siamese = pool_size == strides and pool_size[0] == pool_size[1]
    even = H.value % pool_size[0] == 0 and W.value % pool_size[1] == 0

    if siamese and even:
        pooler = _avgpool2d_reshape_reduce
    else:
        pooler = _avgpool2d_im2col_reduce

    with tf.name_scope('avgpool2d'):
        with tf.device(prot.server_0.device_name):
            y_on_0 = pooler(x_on_0, pool_size, strides, padding)

        with tf.device(prot.server_1.device_name):
            y_on_1 = pooler(x_on_1, pool_size, strides, padding)

        return y_on_0, y_on_1, scalar


def _avgpool2d_reshape_reduce(x: BackingTensor,
                              pool_size: Tuple[int, int],
                              strides: Tuple[int, int],
                              padding: str) -> BackingTensor:
    pool_height, pool_width = tf.Dimension(pool_size[0]), tf.Dimension(pool_size[1])
    N, C, H, W = x.shape
    x_reshaped = x.reshape([N,
                            C,
                            H // pool_height,
                            pool_height,
                            W // pool_width,
                            pool_width])
    return x_reshaped.sum(axis=3).sum(axis=4)


def _avgpool2d_im2col_reduce(x: BackingTensor,
                             pool_size: Tuple[int, int],
                             strides: Tuple[int, int],
                             padding: str) -> BackingTensor:
    batch, channels, height, width = x.shape
    pool_height, pool_width = pool_size

    if padding == "SAME":
        out_height: int = math.ceil(int(height) / strides[0])
        out_width: int = math.ceil(int(width) / strides[1])
    else:
        out_height = math.ceil((int(height) - pool_size[0] + 1) / strides[0])
        out_width = math.ceil((int(width) - pool_size[1] + 1) / strides[1])

    x_split = x.reshape((batch * channels, 1, height, width))
    x_cols = x_split.im2col(pool_height, pool_width, padding, strides[0])
    x_cols_sum = x_cols.sum(axis=0)
    out = x_cols_sum.reshape([out_height, out_width, batch, channels]).transpose([2, 3, 0, 1])

    return out


def _avgpool2d_public(prot: Pond,
                      x: PondPublicTensor,
                      pool_size: Tuple[int, int],
                      strides: Tuple[int, int],
                      padding: str) -> PondPublicTensor:
    y_on_0, y_on_1, scalar = _avgpool2d_core(prot, x, pool_size, strides, padding)

    with tf.name_scope('avgpool2d'):
        return PondPublicTensor(prot, y_on_0, y_on_1) * scalar


def _avgpool2d_private(prot: Pond,
                       x: PondPrivateTensor,
                       pool_size: Tuple[int, int],
                       strides: Tuple[int, int],
                       padding: str) -> PondPrivateTensor:
    y_on_0, y_on_1, scalar = _avgpool2d_core(prot, x, pool_size, strides, padding)

    with tf.name_scope('avgpool2d'):
        return PondPrivateTensor(prot, y_on_0, y_on_1) * scalar


def _avgpool2d_masked(prot: Pond,
                      x: PondMaskedTensor,
                      pool_size: Tuple[int, int],
                      strides: Tuple[int, int],
                      padding: str) -> PondPrivateTensor:
    y_on_0, y_on_1, scalar = _avgpool2d_core(prot, x.unmasked, pool_size, strides, padding)

    with tf.name_scope('avgpool2d'):
        return PondPrivateTensor(prot, y_on_0, y_on_1) * scalar

#
# transpose helpers
#


def _transpose_public(prot, x, perm=None):
    assert isinstance(x, PondPublicTensor)

    x_on_0, x_on_1 = x.unwrapped

    with tf.name_scope('transpose'):

        with tf.device(prot.server_0.device_name):
            x_on_0_t = x_on_0.transpose(perm=perm)

        with tf.device(prot.server_1.device_name):
            x_on_1_t = x_on_1.transpose(perm=perm)

    x_t = PondPublicTensor(prot, x_on_0_t, x_on_1_t)
    return x_t


def _transpose_private(prot, x, perm=None):
    assert isinstance(x, PondPrivateTensor)

    x0, x1 = x.unwrapped

    with tf.name_scope('transpose'):

        with tf.device(prot.server_0.device_name):
            x0_t = x0.transpose(perm=perm)

        with tf.device(prot.server_1.device_name):
            x1_t = x1.transpose(perm=perm)

    x_t = PondPrivateTensor(prot, x0_t, x1_t)
    return x_t


def _transpose_masked(prot, x_masked, perm=None):
    assert isinstance(x_masked, PondMaskedTensor)

    a, a0, a1, alpha_on_0, alpha_on_1 = x_masked.unwrapped

    with tf.name_scope('transpose'):

        with tf.device(prot.crypto_producer.device_name):
            a_t = a.transpose(perm=perm)

        with tf.device(prot.server_0.device_name):
            a0_t = a0.transpose(perm=perm)
            alpha_on_0_t = alpha_on_0.transpose(perm=perm)

        with tf.device(prot.server_1.device_name):
            a1_t = a1.transpose(perm=perm)
            alpha_on_1_t = alpha_on_1.transpose(perm=perm)

    x_unmasked_t = prot.transpose(x_masked.unmasked, perm=perm)
    x_t = PondMaskedTensor(prot, x_unmasked_t, a_t, a0_t, a1_t, alpha_on_0_t, alpha_on_1_t)
    return x_t


#
# strided slice helpers
#


def _strided_slice_public(prot, x: PondPublicTensor, args: Any, kwargs: Any):
    assert isinstance(x, PondPublicTensor)

    x_on_0, x_on_1 = x.unwrapped

    with tf.name_scope('strided_slice'):

        with tf.device(prot.server_0.device_name):
            x_on_0_slice = x_on_0.strided_slice(args, kwargs)

        with tf.device(prot.server_1.device_name):
            x_on_1_slice = x_on_1.strided_slice(args, kwargs)

    x_slice = PondPublicTensor(prot, x_on_0_slice, x_on_1_slice)
    return x_slice


def _strided_slice_private(prot, x: PondPrivateTensor, args: Any, kwargs: Any):
    assert isinstance(x, PondPrivateTensor)

    x0, x1 = x.unwrapped

    with tf.name_scope('strided_slice'):

        with tf.device(prot.server_0.device_name):
            x0_slice = x0.strided_slice(args, kwargs)

        with tf.device(prot.server_1.device_name):
            x1_slice = x1.strided_slice(args, kwargs)

    x_slice = PondPrivateTensor(prot, x0_slice, x1_slice)
    return x_slice


def _strided_slice_masked(prot, x_masked: PondMaskedTensor, args: Any, kwargs: Any):
    assert isinstance(x_masked, PondMaskedTensor)

    a, a0, a1, alpha_on_0, alpha_on_1 = x_masked.unwrapped

    with tf.name_scope('strided_slice'):

        with tf.device(prot.crypto_producer.device_name):
            a_slice = a.strided_slice(args, kwargs)

        with tf.device(prot.server_0.device_name):
            a0_slice = a0.strided_slice(args, kwargs)
            alpha_on_0_slice = alpha_on_0.strided_slice(args, kwargs)

        with tf.device(prot.server_1.device_name):
            a1_slice = a1.strided_slice(args, kwargs)
            alpha_on_1_slice = alpha_on_1.strided_slice(args, kwargs)

    x_unmasked_slice = prot.strided_slice(x_masked.unmasked, args, kwargs)
    x_slice = PondMaskedTensor(prot, x_unmasked_slice, a_slice,
                               a0_slice, a1_slice, alpha_on_0_slice,
                               alpha_on_1_slice)
    return x_slice


#
# strided slice helpers
#


def _stack_public(prot, x: List[PondPublicTensor], axis: int = 0):
    x_on_0 = []
    x_on_1 = []
    for i in x:
        i0, i1 = i.unwrapped
        x_on_0.append(i0)
        x_on_1.append(i1)

    with tf.name_scope('stack'):

        with tf.device(prot.server_0.device_name):
            x_on_0_stack = stack(x_on_0, axis=axis)

        with tf.device(prot.server_1.device_name):
            x_on_1_stack = stack(x_on_1, axis=axis)

    x_stack = PondPublicTensor(prot, x_on_0_stack, x_on_1_stack)
    return x_stack


def _stack_private(prot, x: List[PondPrivateTensor], axis: int = 0):
    x0 = []
    x1 = []
    for i in x:
        i0, i1 = i.unwrapped
        x0.append(i0)
        x1.append(i1)

    with tf.name_scope('stack'):

        with tf.device(prot.server_0.device_name):
            x0_stack = stack(x0, axis=axis)

        with tf.device(prot.server_1.device_name):
            x1_stack = stack(x1, axis=axis)

    x_stack = PondPrivateTensor(prot, x0_stack, x1_stack)
    return x_stack


def _stack_masked(prot, x_masked: List[PondMaskedTensor], axis: int = 0):
    a = []
    a0 = []
    a1 = []
    alpha_on_0 = []
    alpha_on_1 = []
    for i in x_masked:
        ii, i0, i1, i_on_0, i_on_1 = i.unwrapped
        a.append(ii)
        a0.append(i0)
        a1.append(i1)
        alpha_on_0.append(i_on_0)
        alpha_on_1.append(i_on_1)

    with tf.name_scope('stack'):

        with tf.device(prot.crypto_producer.device_name):
            a_stack = stack(a, axis=axis)

        with tf.device(prot.server_0.device_name):
            a0_stack = stack(a0, axis=axis)
            alpha_on_0_stack = stack(alpha_on_0, axis=axis)

        with tf.device(prot.server_1.device_name):
            a1_stack = stack(a1, axis=axis)
            alpha_on_1_stack = stack(alpha_on_1, axis=axis)

    x_unmasked_stack = prot.stack(x_masked.unmasked, axis=axis)
    x_stack = PondMaskedTensor(prot, x_unmasked_stack, a_stack,
                               a0_stack, a1_stack, alpha_on_0_stack,
                               alpha_on_1_stack)
    return x_stack


#
# mask helpers
#


def _mask_private(prot, x: PondPrivateTensor) -> PondMaskedTensor:
    assert isinstance(x, PondPrivateTensor)

    x0, x1 = x.unwrapped
    shape = x.shape

    with tf.name_scope('mask'):

        with tf.device(prot.crypto_producer.device_name):
            a = BackingTensor.sample_uniform(shape)
            a0, a1 = _share(a)

        with tf.device(prot.server_0.device_name):
            alpha0 = x0 - a0

        with tf.device(prot.server_1.device_name):
            alpha1 = x1 - a1

        # exchange of alphas

        with tf.device(prot.server_0.device_name):
            alpha_on_0 = _reconstruct(alpha0, alpha1)

        with tf.device(prot.server_1.device_name):
            alpha_on_1 = _reconstruct(alpha0, alpha1)

    x_masked = PondMaskedTensor(prot, x, a, a0, a1, alpha_on_0, alpha_on_1)
    return x_masked


#
# reshape helpers
#


def _reshape_public(prot, x: PondPublicTensor, shape: List[int]) -> PondPublicTensor:
    assert isinstance(x, PondPublicTensor)

    x_on_0, x_on_1 = x.unwrapped

    with tf.name_scope('reshape'):

        with tf.device(prot.server_0.device_name):
            x_on_0_reshaped = x_on_0.reshape(shape)

        with tf.device(prot.server_1.device_name):
            x_on_1_reshaped = x_on_1.reshape(shape)

    x_reshaped = PondPublicTensor(prot, x_on_0_reshaped, x_on_1_reshaped)
    return x_reshaped


def _reshape_private(prot, x: PondPrivateTensor, shape: List[int]) -> PondPrivateTensor:
    assert isinstance(x, PondPrivateTensor)

    x0, x1 = x.unwrapped

    with tf.name_scope('reshape'):

        with tf.device(prot.server_0.device_name):
            x0_reshaped = x0.reshape(shape)

        with tf.device(prot.server_1.device_name):
            x1_reshaped = x1.reshape(shape)

    x_reshaped = PondPrivateTensor(prot, x0_reshaped, x1_reshaped)
    return x_reshaped


def _reshape_masked(prot, x_masked: PondMaskedTensor, shape: List[int]) -> PondMaskedTensor:
    assert isinstance(x_masked, PondMaskedTensor)

    a, a0, a1, alpha_on_0, alpha_on_1 = x_masked.unwrapped

    with tf.name_scope('reshape'):

        with tf.device(prot.crypto_producer.device_name):
            a_reshaped = a.reshape(shape)

        with tf.device(prot.server_0.device_name):
            a0_reshaped = a0.reshape(shape)
            alpha_on_0_reshaped = alpha_on_0.reshape(shape)

        with tf.device(prot.server_1.device_name):
            a1_reshaped = a1.reshape(shape)
            alpha_on_1_reshaped = alpha_on_1.reshape(shape)

    x_unmasked_reshaped = prot.reshape(x_masked.unmasked)
    x_reshaped = PondMaskedTensor(
        prot, x_unmasked_reshaped, a_reshaped,
        a0_reshaped, a1_reshaped,
        alpha_on_0_reshaped, alpha_on_1_reshaped)

    return x_reshaped


#
# expand dims helpers
#


def _expand_dims_public(prot, x, axis=None):
    assert isinstance(x, PondPublicTensor)

    x_on_0, x_on_1 = x.unwrapped

    with tf.name_scope('expand'):

        with tf.device(prot.server_0.device_name):
            x_on_0_e = x_on_0.expand_dims(axis=axis)

        with tf.device(prot.server_1.device_name):
            x_on_1_e = x_on_1.expand_dims(axis=axis)

    x_e = PondPublicTensor(prot, x_on_0_e, x_on_1_e)
    return x_e


def _expand_dims_private(prot, x, axis=None):
    assert isinstance(x, PondPrivateTensor)

    x0, x1 = x.unwrapped

    with tf.name_scope('expand'):

        with tf.device(prot.server_0.device_name):
            x0_e = x0.expand_dims(axis=axis)

        with tf.device(prot.server_1.device_name):
            x1_e = x1.expand_dims(axis=axis)

    x_e = PondPrivateTensor(prot, x0_e, x1_e)
    return x_e


def _expand_dims_masked(prot, x_masked, axis=None):
    assert isinstance(x_masked, PondMaskedTensor)

    a, a0, a1, alpha_on_0, alpha_on_1 = x_masked.unwrapped

    with tf.name_scope('expand'):

        with tf.device(prot.crypto_producer.device_name):
            a_e = a.expand_dims(axis=axis)

        with tf.device(prot.server_0.device_name):
            a0_e = a0.expand_dims(axis=axis)
            alpha_on_0_e = alpha_on_0.expand_dims(axis=axis)

        with tf.device(prot.server_1.device_name):
            a1_e = a1.expand_dims(axis=axis)
            alpha_on_1_e = alpha_on_1.expand_dims(axis=axis)

    x_unmasked_e = prot.expand_dims(x_masked.unmasked, axis=axis)
    x_e = PondMaskedTensor(prot, x_unmasked_e, a_e, a0_e, a1_e, alpha_on_0_e, alpha_on_1_e)
    return x_e


#
# squeeze helpers
#


def _squeeze_public(prot, x: PondPublicTensor, axis: List[int]) -> PondPublicTensor:
    assert isinstance(x, PondPublicTensor)

    x_on_0, x_on_1 = x.unwrapped

    with tf.name_scope('squeeze'):

        with tf.device(prot.server_0.device_name):
            x_on_0_squeezed = x_on_0.squeeze(axis)

        with tf.device(prot.server_1.device_name):
            x_on_1_squeezed = x_on_1.squeeze(axis)

    x_squeezed = PondPublicTensor(prot, x_on_0_squeezed, x_on_1_squeezed)
    return x_squeezed


def _squeeze_private(prot, x: PondPrivateTensor, axis: List[int]) -> PondPrivateTensor:
    assert isinstance(x, PondPrivateTensor)

    x0, x1 = x.unwrapped

    with tf.name_scope('squeeze'):

        with tf.device(prot.server_0.device_name):
            x0_squeezed = x0.squeeze(axis)

        with tf.device(prot.server_1.device_name):
            x1_squeezed = x1.squeeze(axis)

    x_squeezed = PondPrivateTensor(prot, x0_squeezed, x1_squeezed)
    return x_squeezed


def _squeeze_masked(prot, x_masked: PondMaskedTensor, axis: List[int]) -> PondMaskedTensor:
    assert isinstance(x_masked, PondMaskedTensor)

    a, a0, a1, alpha_on_0, alpha_on_1 = x_masked.unwrapped

    with tf.name_scope('squeeze'):

        with tf.device(prot.crypto_producer.device_name):
            a_squeezed = a.squeeze(axis)

        with tf.device(prot.server_0.device_name):
            a0_squeezed = a0.squeeze(axis)
            alpha_on_0_squeezed = alpha_on_0.squeeze(axis)

        with tf.device(prot.server_1.device_name):
            a1_squeezed = a1.squeeze(axis)
            alpha_on_1_squeezed = alpha_on_1.squeeze(axis)

    x_unmasked_squeezed = prot.squeeze(x_masked.unmasked)
    x_squeezed = PondMaskedTensor(
        prot, x_unmasked_squeezed, a_squeezed,
        a0_squeezed, a1_squeezed,
        alpha_on_0_squeezed, alpha_on_1_squeezed)

    return x_squeezed
