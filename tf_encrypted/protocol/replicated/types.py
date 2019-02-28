from enum import Enum, unique


@unique
class Dtypes(Enum):
    FIXED10 = 1
    FIXED16 = 2
    INTEGER = 3
    REPLICATED3 = 4
