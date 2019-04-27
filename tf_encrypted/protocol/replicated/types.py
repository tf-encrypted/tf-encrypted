from enum import Enum, unique
from .tensor import Tensor


@unique
class Dtypes(Enum):
    FIXED10 = 1
    FIXED16 = 2
    INTEGER = 3
    REPLICATED3 = 4
