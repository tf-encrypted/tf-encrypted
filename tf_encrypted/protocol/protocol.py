"""Base abstraction for a Protocol."""
import functools
from abc import ABC
from types import TracebackType
from typing import Any
from typing import Callable
from typing import Optional

import tensorflow as tf

import tf_encrypted as tfe

from ..config import get_config
from ..tensor.factory import AbstractTensor

nodes = dict()


class Protocol(ABC):
    """
    Protocol is the base class that other protocols in TF Encrypted will extend.

    Do not directly instantiate this class.  You should use a subclass instead,
    such as :class:`~tf_encrypted.protocol.protocol.SecureNN`
    or :class:`~tf_encrypted.protocol.protocol.Pond`
    """

    def __enter__(self) -> "Protocol":
        self.last_protocol = tfe.get_protocol()
        tfe.set_protocol(self)
        return self

    def __exit__(
        self,
        exception_type,
        exception_value: Optional[Exception],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        tfe.set_protocol(self.last_protocol)

    def reset(self):
        pass


class TFETensorBone(ABC):
    pass


class TFETensor(ABC):
    pass


class TFEVariable(TFETensor):
    pass


class TFEPrivateTensor(TFETensor):
    pass


class TFEPrivateVariable(TFEPrivateTensor, TFEVariable):
    pass


class TFEPublicTensor(TFETensor):
    pass


class TFEPublicVariable(TFEPublicTensor, TFEVariable):
    pass


def make_hashable(x):
    if isinstance(x, (tuple, list)):
        return tuple([make_hashable(y) for y in x])
    elif isinstance(x, dict):
        return tuple(
            sorted(
                [(make_hashable(item[0]), make_hashable(item[1])) for item in x.items()]
            )
        )
    elif isinstance(x, tf.TensorShape):
        return tuple(x.as_list())
    else:
        try:
            hash(x)
            return x
        except TypeError:
            # not safe
            # collision may happen
            return str(id(x)) + str(x)


def memoize(func: Callable) -> Callable:
    """
    memoize(func) -> Callable

    Decorates a function for memoization, which explicitly caches the function's
    output.

    :param Callable func: The function to memoize
    """

    @functools.wraps(func)
    def cache_nodes(self: Protocol, *args: Any, **kwargs: Any) -> AbstractTensor:
        if get_config().debug:
            return func(self, *args, **kwargs)

        hashable_args = make_hashable(args)
        hashable_kwargs = make_hashable(kwargs)
        node_key = (func.__name__, hashable_args, hashable_kwargs)

        cached_result = nodes.get(node_key, None)
        if cached_result is not None:
            return cached_result

        result = func(self, *args, **kwargs)

        nodes[node_key] = result
        return result

    return cache_nodes


def unwrap_func(wrap_args):
    if wrap_args is None:
        return None

    if isinstance(wrap_args, (list, tuple)):
        return [unwrap_func(arg) for arg in wrap_args]
    elif isinstance(wrap_args, TFETensor):
        return wrap_args.bone
    elif isinstance(wrap_args, tf.Tensor):
        return wrap_args
    elif isinstance(wrap_args, (float, int, str)):
        return wrap_args
    else:
        raise TypeError("Don't know how to unwrap {}".format(type(wrap_args)))


def wrap_func(unwrap_args):
    if unwrap_args is None:
        return None

    prot = tfe.get_protocol()

    if isinstance(unwrap_args, (list, tuple)):
        return [wrap_func(arg) for arg in unwrap_args]
    elif isinstance(unwrap_args, TFETensorBone):
        return prot.from_bone(unwrap_args)
    elif isinstance(unwrap_args, tf.Tensor):
        return unwrap_args
    elif isinstance(unwrap_args, (float, int, str)):
        return unwrap_args
    else:
        raise TypeError("Don't know how to wrap {}".format(type(unwrap_args)))


def input_unwrap(wrap_args, wrap_kwargs):

    unwrap_args = unwrap_func(wrap_args)
    unwrap_kwargs = {}

    for kwarg in wrap_kwargs.items():
        unwrap_kwargs[kwarg[0]] = unwrap_func(kwarg[1])
    return unwrap_args, unwrap_kwargs


def input_wrap(unwrap_args, unwrap_kwargs):

    wrap_args = wrap_func(unwrap_args)
    wrap_kwargs = {}

    for kwarg in unwrap_kwargs.items():
        wrap_kwargs[kwarg[0]] = wrap_func(kwarg[1])
    return wrap_args, wrap_kwargs


def function(func: Callable) -> Callable:
    """
    Compiling a function into a callable TensorFlow graph.

    Do not use this decorator in every function, just in upper function.
    Because 'tf.function' decorator require function's input and output
    to be tensorflow tensors, so we need to transform tfe tensors to
    tensorflow tensors and transform inversely.
    """

    @tf.function
    def graph_function(args, kwargs):
        args, kwargs = input_wrap(args, kwargs)
        result = func(*args, **kwargs)
        result = unwrap_func(result)
        return result

    @functools.wraps(func)
    def wrap_function(*args: Any, **kwargs: Any) -> AbstractTensor:
        if get_config().debug:
            return func(*args, **kwargs)

        args, kwargs = input_unwrap(args, kwargs)
        result = graph_function(args, kwargs)
        result = wrap_func(result)

        return result

    return wrap_function
