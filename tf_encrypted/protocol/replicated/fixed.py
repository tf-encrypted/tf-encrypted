from .types import Dtypes
from .tensor import Tensor


class Fixed10(Tensor):
    base = 2
    precision_fractional = 10

    def __init__(self, value):
        self._backing = value

    @property
    def dtype(self):
        return Dtypes.FIXED10

    @property
    def shape(self):
        return self.backing.shape

    @property
    def backing(self):
        return self._backing


class Fixed16(Tensor):
    base = 2
    precision_fractional = 16

    def __init__(self, value):
        self._backing = value

    @property
    def dtype(self):
        return Dtypes.FIXED16

    @property
    def shape(self):
        return self._backing.shape

    @property
    def backing(self):
        return self._backing
