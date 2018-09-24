from typing import Any, Dict, Optional, Union, List, Callable, Tuple, Type

from . import nn
from . import errors
from . import python
from . import summary
from . import train
import numpy as np

__all__ = [
    'nn',
    'errors',
    'python',
    'summary',
    'train'
]

GraphElement = Union[
    'Operation',
    'Tensor',
    'SparseTensor',
    str,
]

TFTypes = Union[
    Type['int8'],
    Type['int16'],
    Type['int32'],
    Type['int64'],
    Type['uint8'],
    Type['uint16'],
    Type['uint32'],
    Type['uint64'],
    Type['float16'],
    Type['float32'],
    Type['float64'],
]


class dtype:
    ...


class string(str):
    ...


# Integer types
class number():
    ...


class integer(number):
    ...


class signedinteger(integer):
    ...


class int8(signedinteger):
    ...


class int16(signedinteger):
    ...


class int32(signedinteger):
    ...


class int64(signedinteger):
    ...


class unsignedinteger(integer):
    ...


class uint8(unsignedinteger):
    ...


class uint16(unsignedinteger):
    ...


class uint32(unsignedinteger):
    ...


class uint64(unsignedinteger):
    ...


class floating(number):
    ...


class float16(floating):
    ...


class float32(floating):
    ...


class float64(floating):
    ...


class dtypes:
    float32 = float32

    def as_string(self) -> str:
        ...


class Operation:
    ...


class Tensor:
    op: Operation

    @property
    def shape(self) -> 'TensorShape':
        ...

    def __add__(self, other):
        ...

    def __mul__(self, other):
        ...

    def __sub__(self, other):
        ...

    def __mod__(self, other):
        ...

    def __getitem__(self, slice):
        ...


class SparseTensor:
    ...


class TensorShape:
    def __init__(self, dims: Optional[List[Any]] = None) -> None:
        ...

    def is_fully_defined(self) -> bool:
        ...

    def __len__(self) -> int:
        ...


class ClusterSpec:
    ...


class gpu_options:
    def __init__(self):
        self.allow_growth: bool


class ConfigProto:
    def __init__(self,
                 log_device_placement: bool,
                 allow_soft_placement: bool,
                 device_count: Optional[Dict[str, int]] = None,
                 inter_op_parallelism_threads: Optional[int] = 0,
                 intra_op_parallelism_threads: Optional[int] = 0) -> None:
        ...


class Graph:
    def get_tensor_by_name(self, name: str) -> Tensor:
        ...


class NodeDef:
    @property
    def name(self) -> str:
        ...

    @property
    def op(self) -> str:
        ...

    @property
    def attr(self) -> List['AttrValue']:
        ...


class AttrValue:
    ...


class GraphDef:
    def ParseFromString(self, str: str) -> None:
        ...

    @property
    def node(self) -> List[NodeDef]:
        ...


class RNNCell:
    ...


class BaseSession:
    # TODO: options is of type RunOption, run_metadata is of type RunMetadata
    # Return type is option of:
    # single graph element if fetches is a single graph element  OR
    # list of graph elements if fetches is a list of single graph elements OR
    # a dictionary
    # Leaving it as Any for now
    def run(self, fetches: Any,
            feed_dict: Optional[Dict[Any, Any]] = ...,
            run_options: Any = ...,
            run_metadata: Any = ...,
            options: Optional['RunOptions'] = None,
            ) -> Any:
        ...

    def close(self) -> None:
        ...


class Session(BaseSession):
    def __init__(self,
                 target: str = ...,
                 graph: Optional[Graph] = ...,
                 config: ConfigProto = ...,
                 ) -> None:
        self.graph = graph
        ...

    def __enter__(self):
        ...

    def __exit__(self, type, value, traceback):
        ...

    def close(self) -> None:
        ...

# defined here
# https://github.com/tensorflow/tensorflow/blob/d8f9538ab48e3c677aaf532769d29bc29a05b76e/tensorflow/python/ops/variables.py#L40


class Variable:
    def __init__(self,
                 initial_value: Any = ...,
                 trainable: Optional[bool] = ...,
                 collections: Optional[Any] = ...,
                 validate_shape: Optional[bool] = ...,
                 caching_device: Optional[Any] = ...,
                 name: Optional[str] = ...,
                 variable_def: Optional[Any] = ...,
                 dtype: Optional[Any] = ...,
                 expected_shape: Optional[Any] = ...,
                 import_scope: Optional[str] = ...,
                 constraint: Optional[Any] = ...,
                 ) -> None:
        ...

    def read_value(self) -> Tensor:
        ...

    @property
    def initializer(self) -> Operation:
        ...


class RunOptions:

    NO_TRACE = 0
    SOFTWARE_TRACE = 1
    HARDWARE_TRACE = 2
    FULL_TRACE = 3

    def __init__(self, trace_level: int) -> None:
        ...


class RunMetadata:
    def __init__(self) -> None:
        self.step_stats: Any


class FIFOQueue:
    def __init__(
        self,
        capacity: int,
        dtypes: List[Any],
        shapes: Optional[List[Any]] = None,
        names: Optional[List[str]] = None,
        shared_name: Optional[str] = None,
        name: Optional[str] = 'fifo_queue'
    ) -> None:
        ...

    def enqueue(self, vals: Any, name: Optional[str] = None) -> Operation:
        ...

    def dequeue(self, name: Optional[str] = None) -> Any:
        ...


# Original function definition for edit_distance here:
# https://github.com/tensorflow/tensorflow/blob/faff6f2a60a01dba57cf3a3ab832279dbe174798/tensorflow/python/ops/array_ops.py#L2049
# return type is Tensor
def edit_distance(hypothesis: Any,
                  truth: Any,
                  normalize: Optional[bool] = ...,
                  name: Optional[str] = ...,
                  ) -> Any:
    ...

# Original function definition for global_variables_initializer here:
# https://github.com/tensorflow/tensorflow/blob/28340a4b12e286fe14bb7ac08aebe325c3e150b4/tensorflow/python/ops/variables.py#L1565


def global_variables_initializer() -> Any:
    ...

# Original function definition for reset_default_graph here:
# https://github.com/tensorflow/tensorflow/blob/28340a4b12e286fe14bb7ac08aebe325c3e150b4/tensorflow/python/framework/ops.py#L5531


def reset_default_graph() -> Graph:
    ...


# Original function definition for placeholder here:
# https://github.com/tensorflow/tensorflow/blob/28340a4b12e286fe14bb7ac08aebe325c3e150b4/tensorflow/python/ops/array_ops.py#L1693
# TODO: improve types
def placeholder(dtype: Any,
                shape: Any = ...,
                name: Optional[str] = ...,
                ) -> Tensor:
    ...

# Original function definition for sparse_placeholder here:
# https://github.com/tensorflow/tensorflow/blob/28340a4b12e286fe14bb7ac08aebe325c3e150b4/tensorflow/python/ops/array_ops.py#L1749
# TODO: improve types


def sparse_placeholder(dtype: Any,
                       shape: Any = ...,
                       name: Optional[str] = ...,
                       ) -> Any:
    ...

# Original function definition for sparse_tensor_to_dense here:
# https://github.com/tensorflow/tensorflow/blob/d8f9538ab48e3c677aaf532769d29bc29a05b76e/tensorflow/python/ops/sparse_ops.py#L948
# sp_input is SparseTensor
# returns Tensor


def sparse_tensor_to_dense(sp_input: Any,
                           default_value: Any = ...,
                           validate_indices: bool = ...,
                           name: Optional[str] = ...,
                           ) -> Any:
    ...

# Original function definition for shape here:
# https://github.com/tensorflow/tensorflow/blob/28340a4b12e286fe14bb7ac08aebe325c3e150b4/tensorflow/python/ops/array_ops.py#L197
# TODO: improve types. return type of None here is a hack
# input is `Tensor` or `SparseTensor`
# out_type is an optional integral data-type (`int32` or `int64`).
# returns a `Tensor` of type specified by `out_type`


def shape(input: Any,
          name: Optional[str] = ...,
          out_type: Any = ...,
          ) -> Any:
    ...

# Original function definition for truncated_normal here:
# https://github.com/tensorflow/tensorflow/blob/70cd9ed2d2ea37a6da6f813a99b32c03e90736a4/tensorflow/python/ops/random_ops.py#L139


def truncated_normal(shape: Any,
                     mean: Any = ...,  # default 0.0
                     stddev: Any = ...,  # default 1.0
                     dtype: Any = dtypes.float32,
                     seed: Any = ...,
                     name: Optional[str] = ...,
                     ) -> Any:
    ...

# Original function definition for reduce_mean here:
# https://github.com/tensorflow/tensorflow/blob/3f8febf04b075eef0950a18c7e122f0addeacfe9/tensorflow/python/ops/math_ops.py#L1384
# Returns Tensor


def reduce_mean(input_tensor: Tensor,
                axis: Any = ...,
                keepdims: Any = ...,
                name: Optional[str] = ...,
                reduction_indices: Any = ...,
                keep_dims: Any = ...,
                ) -> Any:
    ...


def name_scope(name: str) -> Any:
    ...


def device(name: str) -> Any:
    ...


def random_normal(
    shape: List[int],
    mean: Any = ...,
    stddev: Any = ...,
    dtype: Any = dtypes.float32,
    seed: int = ...,
    name: Optional[str] = ...,
) -> Tensor:
    ...


def Print(
    input_: Tensor,
    data: List[Tensor],
    message: Optional[str] = None,
    first_n: Optional[int] = None,
    summarize: Optional[int] = None,
    name: Optional[str] = None
) -> Tensor:
    ...


def group(
    *inputs: Tensor,
    **kwargs: Any,
) -> Operation:
    ...


def reshape(
    tensor: Any,
    shape: Union[Tensor, List[int]],
    name: Optional[str] = None
) -> Tensor:
    ...


def control_dependencies(control_inputs: List[Union[Operation, Tensor]]) -> Any:
    ...


def get_default_graph() -> Graph:
    ...


def while_loop(
    cond: Callable,
    body: Callable,
    loop_vars: Any,
    shape_invariants: Optional[Any] = None,
    parallel_iterations: int = 10,
    back_prop: bool = True,
    swap_memory: bool = False,
    name: Optional[str] = None,
    maximum_iterations: Optional[int] = None,
    return_same_structure: bool = False
) -> Any:
    ...


def identity(input: Tensor, name: Optional[str] = None) -> Tensor:
    ...


def constant(
    value: Union[Any, List[Any]],
    dtype: Optional[Any] = None,
    shape: Optional[List[int]] = None,
    name: Optional[str] = 'Const',
    verify_shape: bool = False,
) -> Tensor:
    ...


def matmul(
    a: Union[np.ndarray, Tensor, Variable],
    b: Union[np.ndarray, Tensor, Variable],
    transpose_a=False,
    transpose_b=False,
    adjoint_a=False,
    adjoint_b=False,
    a_is_sparse=False,
    b_is_sparse=False,
    name=None
) -> Tensor:
    ...


def random_uniform(
    shape: Union[List[int], Tuple[int, ...], TensorShape],
    minval: Any = 0,
    maxval: Optional[Any] = None,
    dtype: Optional[TFTypes] = float32,
    seed: Optional[int] = None,
    name: Optional[str] = None
) -> Tensor:
    ...


def zeros(
    shape: Union[List[int], Tuple[int, ...], TensorShape],
    dtype: Optional[TFTypes] = float32,
    name: Optional[str] = None
) -> Tensor:
    ...


def assign(
    ref: Variable,
    value: Union[Tensor, np.ndarray],
    validate_shape: Optional[bool] = None,
    use_locking: Optional[bool] = None,
    name: Optional[str] = None
) -> Tensor:
    ...


def transpose(
    a: Union[np.ndarray, Tensor],
    perm: Optional[Union[List[int], Tuple[int, ...]]]=None,
    name: str='transpose',
    conjugate: bool=False
) -> Tensor:
    ...


def stack(
    values: List[Union[np.ndarray, Tensor]],
    axis: int=0,
    name: str='stack'
) -> Tensor:
    ...


def cast(
    x: Union[np.ndarray, Tensor],
    dtype: Optional[TFTypes] = float32,
    name: Optional[str] = None
) -> Tensor:
    ...


def expand_dims(
    input: Tensor,
    axis: int=0,
    name: Optional[str] = None,
    dim: Optional[int] = None
) -> Tensor:
    ...
