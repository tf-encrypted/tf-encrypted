from .pond import PondPublicVariable, PondPrivateVariable
from tensorflow import Variable as tfVariable
from typing import Union, NewType

TFEVariable = Union[PondPublicVariable, PondPrivateVariable, tfVariable]
