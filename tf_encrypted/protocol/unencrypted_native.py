
from ..protocol import Protocol
import abc

from typing import Tuple, List, Union, Optional, Any, NewType, Callable

import tensorflow as tf

from ..tensor.factory import (
    AbstractFactory,
    AbstractTensor,
    AbstractConstant,
    AbstractVariable,
    AbstractPlaceholder,
)

import numpy as np

class UnencryptedNative(Protocol):

    
    def __init__(self,
                 server_0: Optional[Player] = None,
                 tensor_factory: Optional[AbstractFactory] = None,
                 )->None:
        
        
        print(" Note: You are currently using unencrypted Protocol. This could lead to insecure behaviour.")
        
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

        v = value

        with tf.name_scope("constant{}".format("-" + name if name else "")):

            with tf.device(self.server_0.device_name):
                x_on_0 = factory.constant(v)


        return x_on_0
    
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


        return PondPublicPlaceholder(self, x_on_0, x_on_1, apply_scaling)
    
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

            if isinstance(x,UnencryptedTensor):
                return x

            raise TypeError("Don't know how to lift {}".format(type(x)))

        else:

            if isinstance(x, (int, float)):

                if isinstance(y, (int, float)):
                    x = self.define_constant(np.array([x]))
                    y = self.define_constant(np.array([y]))
                    return x, y

                if isinstance(y,UnencryptedTensor):
                    x = self.define_constant(
                        np.array([x]),
                        apply_scaling=apply_scaling or y.is_scaled,
                        factory=y.backing_dtype,
                    )
                    return x, y

                raise TypeError(
                    "Don't know how to lift {}, {}".format(type(x), type(y))
                )

            if isinstance(x,UnencryptedTensor):
                if isinstance(y, (int, float)):
                    y = self.define_constant(
                        np.array([y]),
                        apply_scaling=apply_scaling or x.is_scaled,
                        factory=x.backing_dtype,
                    )
                    return x, y

                if isinstance(y,UnencryptedTensor):
                    return x, y

                raise TypeError(
                    "Don't know how to lift {}, {}".format(type(x), type(y))
                )

            raise TypeError("Don't know how to lift {}, {}".format(type(x), type(y)))
    
    def add(self, x, y):
        """
        add(x, y) -> PondTensor

        Adds two tensors `x` and `y`.

        :param PondTensor x: The first operand.
        :param PondTensor y: The second operand.
        """
        x, y = self.lift(x, y)
        return self.dispatch("add", x, y)
    
class UnencryptedTensor(abc.ABC):
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

    def expand_dims(self):
        """
        :See: tf.expand_dims

        :return: A new PondTensor
        :rtype: PondTensor
        """
        return self.prot.expand_dims(self)

    def reshape(self, shape: List[int]) -> "PondTensor":
        """
        :See: tf.reshape

        :param List[int] shape: The new shape of the tensor.
        :rtype: PondTensor
        :returns: A new tensor with the contents of this tensor, but with the new specified shape.
        """
        return self.prot.reshape(self, shape)

    def reduce_max(self, axis: int) -> "PondTensor":
        """
        :See: tf.reduce_max

        :param int axis: The axis to take the max along
        :rtype: PondTensor
        :returns: A new pond tensor with the max value from each axis.
        """
        return self.prot.reduce_max(self, axis)
