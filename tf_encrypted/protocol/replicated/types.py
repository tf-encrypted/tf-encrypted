from enum import Enum, unique
from .tensor import Tensor


@unique
class Dtypes(Enum):
    FIXED10 = 1
    FIXED16 = 2
    INTEGER = 3
    REPLICATED3 = 4


fixed_config = {
    Dtypes.FIXED10: {
        "base": 2,
        "bits": 10,
    },
    Dtypes.FIXED16: {
        "base": 2,
        "bits": 16
    }
}


class Fixed(Tensor):
    def __init__(self, backing, dtype):
        self._backing = backing
        self._dtype = dtype

        # check to make sure its fixed10 or fixed16

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._backing.shape

    @property
    def backing(self):
        return self._backing
