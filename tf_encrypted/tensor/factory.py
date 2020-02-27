"""Abstract classes for factories and their components."""
import abc
from typing import Optional
from typing import Union

import numpy as np
import tensorflow as tf


class AbstractTensor(abc.ABC):
    """
  An abstraction to use when building tensor classes and subclasses
  in factories.
  """

    @property
    @abc.abstractproperty
    def factory(self):
        pass

    @property
    @abc.abstractproperty
    def shape(self):
        pass

    @property
    @abc.abstractproperty
    def support(self):
        pass

    @abc.abstractmethod
    def identity(self):
        pass


# pylint: disable=abstract-method


class AbstractConstant(AbstractTensor):
    pass


class AbstractPlaceholder(AbstractTensor):
    pass


class AbstractVariable(AbstractTensor):
    pass


# pylint: enable=abstract-method


class AbstractFactory(abc.ABC):
    """An abstraction to use when building tensor factories."""

    @property
    @abc.abstractmethod
    def modulus(self) -> int:
        """The modulus used by this data type."""

    @property
    @abc.abstractmethod
    def native_type(self):
        """The underlying TensorFlow dtype used by this data type."""

    @abc.abstractmethod
    def tensor(self, value: Union[tf.Tensor, np.ndarray]):
        """Wrap raw `value` in this data type as a tensor."""

    @abc.abstractmethod
    def constant(self, value: np.ndarray):
        """Create a constant of this data type using raw `value`."""

    @abc.abstractmethod
    def variable(self, initial_value):
        """Create a variable of this data type using raw `initial_value`."""

    @abc.abstractmethod
    def placeholder(self, shape):
        """Create a placeholder of this data type."""

    @abc.abstractmethod
    def sample_uniform(
        self, shape, minval: Optional[int] = None, maxval: Optional[int] = None
    ):
        """Sample uniform random value of this data type."""

    @abc.abstractmethod
    def sample_bounded(self, shape, bitlength: int):
        """Sample uniform random value of this data type."""

    @abc.abstractmethod
    def stack(self, xs: list, axis: int = 0):
        """Stack tensors of this data type together."""

    @abc.abstractmethod
    def concat(self, xs: list, axis: int):
        """Concatenate tensors of this data type together."""
