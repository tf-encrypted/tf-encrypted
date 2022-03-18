import abc
from typing import Tuple, List, Union, Optional, Callable
from ...tensor.factory import (
    AbstractTensor,
    AbstractConstant,
    AbstractPlaceholder
)
import numpy as np
import tensorflow as tf


class ShareType:
    PUBLIC = None
    ARITHMETIC = "ARITHMETIC"
    BOOLEAN = "BOOLEAN"

    @staticmethod
    def is_legal_type(share_type):
        return (share_type == ShareType.PUBLIC) \
                or (share_type == ShareType.ARITHMETIC) \
                or (share_type == ShareType.BOOLEAN)


#
# Classes representing the base values in the ABY3 protocol.
#


class ABY3Tensor(abc.ABC):
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

    def __init__(self, prot, is_scaled, share_type):
        self.prot = prot
        self.is_scaled = is_scaled
        self.share_type = share_type

    def __repr__(self) -> str:
        return "{}(shape={}, share_type={})".format(type(self).__name__, self.shape, self.share_type)

    def is_arithmetic(self) -> bool:
        return self.share_type == ShareType.ARITHMETIC

    def is_boolean(self) -> bool:
        return self.share_type == ShareType.BOOLEAN

    def is_public(self) -> bool:
        return self.share_type == ShareType.PUBLIC

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
        Add `other` to this ABY3Tensor.  This can be another tensor with the same
        backing or a primitive.

        This function returns a new ABY3Tensor and does not modify this one.

        :param ABY3Tensor other: a or primitive (e.g. a float)
        :return: A new ABY3Tensor with `other` added.
        :rtype: ABY3Tensor
        """
        if self.share_type == ShareType.ARITHMETIC or self.share_type == ShareType.PUBLIC:
            return self.prot.add(self, other)
        else:
            raise ValueError("unsupported share type for add: {}".format(self.share_type))

    def __add__(self, other):
        """
        See :meth:`~tf_encrypted.protocol.aby3.ABY3Tensor.add`
        """
        return self.add(other)

    def __radd__(self, other):
        return self + other

    def reduce_sum(self, axis=None, keepdims=False):
        """
        Like :meth:`tensorflow.reduce_sum`

        :param int axis:  The axis to reduce along
        :param bool keepdims: If true, retains reduced dimensions with length 1.
        :return: A new ABY3Tensor
        :rtype: ABY3Tensor
        """
        return self.prot.reduce_sum(self, axis, keepdims)

    def sum(self, axis=None, keepdims=False):
        """
        See :meth:`ABY3Tensor.reduce_sum`
        """
        return self.reduce_sum(axis, keepdims)

    def sub(self, other):
        """
        Subtract `other` from this tensor.

        :param ABY3Tensor other: to subtract
        :return: A new ABY3Tensor
        :rtype: ABY3Tensor
        """
        if self.share_type == ShareType.ARITHMETIC or self.share_type == ShareType.PUBLIC:
            return self.prot.sub(self, other)
        else:
            raise ValueError("unsupported share type for sub: {}".format(self.share_type))

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        if self.share_type == ShareType.ARITHMETIC or self.share_type == ShareType.PUBLIC:
            return self.prot.sub(other, self)
        else:
            raise ValueError("unsupported share type for sub: {}".format(self.share_type))

    def mul(self, other):
        """
        Multiply this tensor with `other`

        :param ABY3Tensor other: to multiply
        :return: A new ABY3Tensor
        :rtype: ABY3Tensor
        """
        return self.prot.mul(self, other)

    def __mul__(self, other):
        return self.prot.mul(self, other)

    def __rmul__(self, other):
        return self.prot.mul(other, self)

    def __truediv__(self, other):
        return self.prot.div(self, other)

    def __mod__(self, other):
        return self.prot.mod(self, other)

    def __pow__(self, p):
        return self.prot.pow(self, p)

    def square(self):
        """
        Square this tensor.

        :return: A new ABY3Tensor
        :rtype: ABY3Tensor
        """
        return self.prot.square(self)

    def matmul(self, other):
        """
        MatMul this tensor with `other`.  This will perform matrix multiplication,
        rather than elementwise like
        :meth:`~tf_encrypted.protocol.aby3.ABY3Tensor.mul`

        :param ABY3Tensor other: to mul
        :return: A new ABY3Tensor
        :rtype: ABY3Tensor
        """
        return self.prot.matmul(self, other)

    def dot(self, other):
        """
        :return: A new ABY3Tensor
        :rtype: ABY3Tensor
        """
        return self.matmul(other)

    def __getitem__(self, slc):
        return self.prot.indexer(self, slc)

    def transpose(self, perm=None):
        """
        Transpose this tensor.

        See :meth:`tensorflow.transpose`

        :param List[int]: A permutation of the dimensions of this tensor.

        :return: A new ABY3Tensor
        :rtype: ABY3Tensor
        """
        return self.prot.transpose(self, perm)

    def truncate(self, trunc_type="trunc2"):
        """
        Truncate this tensor.

        :return: A new ABY3Tensor
        :rtype: ABY3Tensor
        """
        return self.prot.truncate(self, trunc_type)

    def expand_dims(self, axis=None):
        """
        :See: tf.expand_dims

        :return: A new ABY3Tensor
        :rtype: ABY3Tensor
        """
        return self.prot.expand_dims(self, axis=axis)

    def reshape(self, shape: List[int]) -> "ABY3Tensor":
        """
        :See: tf.reshape

        :param List[int] shape: The new shape of the tensor.
        :rtype: ABY3Tensor
        :returns: A new tensor with the contents of this tensor, but with the new
            specified shape.
        """
        return self.prot.reshape(self, shape)

    def __neg__(self):
        return self.prot.negative(self)

    def negative(self) -> "ABY3Tensor":
        """
        :See: tf.negative

        :rtype: ABY3Tensor
        :returns: A new tensor with numerical negative value element-wise computed.
        """
        return self.prot.negative(self)

    def reduce_max(self, axis: int) -> "ABY3Tensor":
        """
        :See: tf.reduce_max

        :param int axis: The axis to take the max along
        :rtype: ABY3Tensor
        :returns: A new ABY3 tensor with the max value from each axis.
        """
        return self.prot.reduce_max(self, axis)

    def bitwise_xor(self, other):
        if self.share_type == ShareType.BOOLEAN:
            return self.prot.xor(self, other)
        else:
            raise ValueError("Unsupported share type for xor: {}".format(self.share_type))

    def __xor__(self, other):
        return self.bitwise_xor(other)

    def bitwise_and(self, other):
        if self.share_type == ShareType.BOOLEAN:
            return self.prot.and_(self, other)
        else:
            raise ValueError("unsupported share type for and: {}".format(self.share_type))

    def __and__(self, other):
        return self.bitwise_and(other)

    def bitwise_or(self, other):
        if self.share_type == ShareType.BOOLEAN:
            return self.prot.or_(self, other)
        else:
            raise ValueError("unsupported share type for and: {}".format(self.share_type))

    def __or__(self, other):
        return self.bitwise_or(other)

    def invert(self):
        if self.share_type == ShareType.BOOLEAN:
            return self.prot.not_(self)
        else:
            raise ValueError("unsupported share type for and: {}".format(self.share_type))

    def __invert__(self):
        return self.invert()

    def __lt__(self, other):
        return self.prot.less_than(self, other)

    def __le__(self, other):
        return self.prot.less_equal(self, other)

    def __gt__(self, other):
        return self.prot.greater_than(self, other)

    def __ge__(self, other):
        return self.prot.greater_equal(self, other)

    def __lshift__(self, steps):
        return self.prot.lshift(self, steps)

    def lshift(self, steps):
        return self.prot.lshift(self, steps)

    def __rshift__(self, steps):
        return self.prot.rshift(self, steps)

    def rshift(self, steps):
        return self.prot.rshift(self, steps)

    def arith_rshift(self, steps):
        return self.rshift(steps)

    def logical_rshift(self, steps):
        return self.prot.logical_rshift(self, steps)

    def write(self, filename_prefix):
        return self.prot.write(self, filename_prefix)

    def cast(self, factory):
        return self.prot.cast(self, factory)


class ABY3PublicTensor(ABY3Tensor):
    """
    This class represents a public tensor, known by at least by the three servers
    but potentially known by more. Although there is only a single value we
    replicate it on both servers to avoid sending it from one to the other
    in the operations where it's needed by both (eg multiplication).
    """

    dispatch_id = "public"

    def __init__(
            self,
            prot: "ABY3",
            values: List[AbstractTensor],
            is_scaled: bool
    ) -> None:
        assert all(isinstance(v, AbstractTensor) for v in values)
        assert all((v.shape == values[0].shape) for v in values)

        super(ABY3PublicTensor, self).__init__(prot, is_scaled, ShareType.PUBLIC)
        self.values = values

    def __repr__(self) -> str:
        return "ABY3PublicTensor(shape={}, share_type={})".format(self.shape, self.share_type)

    @property
    def shape(self) -> List[int]:
        return self.values[0].shape

    @property
    def backing_dtype(self):
        return self.values[0].factory

    @property
    def unwrapped(self) -> Tuple[AbstractTensor, ...]:
        """
        Unwrap the tensor.

        This will return the value for each of the parties that collectively own
        the tensor.

        In most cases, this will be the same value on each device.

        .. code-block:: python

            x_0, y_0, z_0 = tensor.unwrapped
            # x_0 == 10 with the value pinned to player_0's device.
            # y_0 == 10 with the value pinned to player_1's device.
            # z_0 == 10 with the value pinned to player_2's device.

        In most cases you will want to work on this data on the specified device.

        .. code-block:: python

            x_0, y_0, z_0= tensor.unwrapped

            with tf.device(prot.player_0.device_name):
                # act on x_0

            with tf.device(prot.player_1.device_name):
                # act on y_0

            with tf.device(prot.player_2.device_name):
                # act on z_0

        In most cases you will not need to use this method.  All funtions
        will hide this functionality for you (e.g. `add`, `mul`, etc).
        """
        return self.values

    def decode(self) -> Union[np.ndarray, tf.Tensor]:
        return self.prot._decode(self.values[0], self.is_scaled)  # pylint: disable=protected-access

    def to_native(self):
        return self.decode()


class ABY3Constant(ABY3PublicTensor):
    """
    This class essentially represents a public value, however it additionally
    records the fact that the underlying value was declared as a constant.
    """

    def __init__(self, prot, constants, is_scaled):
        assert all(isinstance(c, AbstractConstant) for c in constants)
        assert all((c.shape == constants[0].shape) for c in constants)

        super(ABY3Constant, self).__init__(
            prot, constants, is_scaled
        )
        self.constants = constants

    def __repr__(self) -> str:
        return "ABY3Constant(shape={}, share_type={})".format(self.shape, self.share_type)

class ABY3PrivateTensor(ABY3Tensor):
    """
    This class represents a private value that may be unknown to everyone.
    """

    dispatch_id = "private"

    def __init__(self, prot, shares, is_scaled, share_type):
        assert len(shares) == 3
        assert all((ss.shape == shares[0][0].shape) for s in shares for ss in s), "Shares have different shapes."

        super(ABY3PrivateTensor, self).__init__(prot, is_scaled, share_type)
        self.shares = shares

    def __repr__(self) -> str:
        return "ABY3PrivateTensor(shape={}, share_type={})".format(self.shape, self.share_type)

    @property
    def shape(self) -> List[int]:
        return self.shares[0][0].shape

    @property
    def backing_dtype(self):
        return self.shares[0][0].factory

    @property
    def unwrapped(self):
        return self.shares

    def reveal(self) -> ABY3PublicTensor:
        return self.prot.reveal(self)


class ABY3PrivateVariable(ABY3PrivateTensor):
    """
    This class essentially represents a private value, however it additionally
    records the fact that the backing tensor was declared as a variable in
    order to allow treating it as a variable itself.
    """

    def __init__(self, prot, shares, is_scaled, share_type):

        super(ABY3PrivateVariable, self).__init__(
            prot, shares, is_scaled, share_type
        )
        self.shares = shares
        self.initializer = tf.group(
            *[var.initializer for share in shares for var in share]
        )

    def __repr__(self) -> str:
        return "ABY3PrivateVariable(shape={}, share_type={})".format(self.shape, self.share_type)


class ABY3PrivatePlaceholder(ABY3PrivateTensor):
    """
  This class essentially represents a private value, however it additionally
  records the fact that the backing tensor was declared as a placeholder in
  order to allow treating it as a placeholder itself.
  """

    def __init__(self, prot, shares, is_scaled, share_type):
        assert all(isinstance(ss, AbstractPlaceholder) for s in shares for ss in s), "Shares should be AbstractPlaceholder."

        super().__init__(prot, shares, is_scaled, share_type)
        self.shares = shares

    def __repr__(self) -> str:
        return "PondPrivatePlaceholder(shape={})".format(self.shape)

    def feed(self, value):
        """
    Feed `value` to placeholder
    """
        assert isinstance(value, np.ndarray), type(value)
        enc = self.prot._encode(value, self.is_scaled)
        assert isinstance(enc, np.ndarray)

        # x0, x1 = self.prot._share(enc)
        # assert isinstance(x0, np.ndarray), type(x0)
        # assert isinstance(x1, np.ndarray), type(x1)

        # TODO(Morten)
        #
        # This is a huge hack and it would be better to use `_share` as above.
        # However, _share currently expects its inputs to be TFE tensors backed
        # by tf.Tensors in order to have extra information attached, and not sure
        # we should change this until we've least considered what will happen with
        # TF2 and eager mode.
        #
        # So, to ensure that feeding can be done locally *outside* the TF graph,
        # in the mean time we manually share values here, avoiding a call to
        # `factory.tensor` as that's where tensors are converted to tf.Tensors.
        shape = self.shape
        minval = self.backing_dtype.min
        maxval = self.backing_dtype.max
        # TODO(Morten) not using secure randomness here; reconsider after TF2
        x0 = np.array(
            [random.randrange(minval, maxval) for _ in range(np.product(shape))]
        ).reshape(shape)
        x1 = np.array(
            [random.randrange(minval, maxval) for _ in range(np.product(shape))]
        ).reshape(shape)
        if self.share_type == ShareType.ARITHMETIC:
            x2 = enc - x0 - x1
        else:
            x2 = enc ^ x0 ^ x1

        feed00 = self.shares[0][0].feed(x0)
        feed01 = self.shares[0][1].feed(x1)
        feed10 = self.shares[1][0].feed(x1)
        feed11 = self.shares[1][1].feed(x2)
        feed20 = self.shares[2][0].feed(x2)
        feed21 = self.shares[2][1].feed(x0)
        return {**feed00, **feed01, **feed10, **feed11, **feed20, **feed21}
