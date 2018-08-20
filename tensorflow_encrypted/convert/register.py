import tensorflow as tf

from typing import Any, Dict, List
from ..layers import Conv2D, Relu, Sigmoid, Dense
import numpy as np
import array

from ..protocol.protocol import get_protocol


def register() -> Dict[str, Any]:
    reg = {
        'Placeholder': placeholder,
        'Const': constant,
        'Conv2D': conv2d,
        'Relu': relu,
        'Sigmoid': sigmoid,
        'MatMul': matmul,
        'Shape': shape,
        'StridedSlice': strided_slice,
        # 'Pack': pack,
        # 'Reshape': reshape,
        # 'BiasAdd': bias_add,
        # 'MaxPool': maxpool,
    }

    return reg


def placeholder(node: Any, inputs: List[str], output_lookup: Dict[str, Any]) -> Any:
    return tf.placeholder(node.attr["dtype"].type,
                          shape=node.attr["shape"].shape)


def constant(node: Any, inputs: List[str], output_lookup: Dict[str, Any]) -> Any:
    # need to able to access the underlying weights return the node
    return node


def matmul(node: Any, inputs: List[str], output_lookup: Dict[str, Any]) -> Any:
    a = output_lookup[inputs[0]]
    b = output_lookup[inputs[1]]

    tensor = b.attr["value"].tensor

    b_shape = [i.size for i in tensor.tensor_shape.dim]

    layer = Dense(b_shape[0], b_shape[1])

    dtype = tensor.dtype

    if dtype == tf.float32:
        nums = array.array('f', tensor.tensor_content)
    elif dtype == tf.float64:
        nums = array.array('d', tensor.tensor_content)
    else:
        raise TypeError("Unsupported dtype for weights")

    layer.initialize(initial_weights=np.array(nums).reshape(b_shape))

    return layer.forward(a)


def conv2d(node: Any, inputs: List[str], output_lookup: Dict[str, Any]) -> Any:
    input = output_lookup[inputs[0]]
    filter = output_lookup[inputs[1]]

    shape = [i.size for i in filter.attr["value"].tensor.tensor_shape.dim]
    dtype = filter.attr["dtype"].type
    format = node.attr["data_format"].s
    if format == "NHWC":
        raise AttributeError("Wrong data format for convolution only support NCHW for now")

    layer = Conv2D(shape, strides=int(node.attr["strides"].list.i[0]),
                   padding=node.attr["padding"].s.decode('ascii'))

    if dtype == tf.float32:
        nums = array.array('f', filter.attr["value"].tensor.tensor_content)
    elif dtype == tf.float64:
        nums = array.array('d', filter.attr["value"].tensor.tensor_content)
    else:
        raise TypeError("Unsupported dtype for weights")

    layer.initialize(input.shape, initial_weights=np.array(nums).reshape(shape))

    return layer.forward(input)


def relu(node: Any, inputs: List[str], output_lookup: Dict[str, Any]) -> Any:
    input = output_lookup[inputs[0]]

    return Relu().forward(input)


def sigmoid(node: Any, inputs: List[str], output_lookup: Dict[str, Any]) -> Any:
    input = output_lookup[inputs[0]]

    return Sigmoid().forward(input)


def strided_slice(node: Any, inputs: List[str], output_lookup: Dict[str, Any]) -> Any:
    input = output_lookup[inputs[0]]
    begin = output_lookup[inputs[1]]
    end = output_lookup[inputs[2]]
    strides = output_lookup[inputs[3]]

    begin_mask = node.attr["begin_mask"].i
    end_mask = node.attr["end_mask"].i
    ellipsis_mask = node.attr["ellipsis_mask"].i
    new_axis_mask = node.attr["new_axis_mask"].i
    shrink_axis_mask = node.attr["shrink_axis_mask"].i

    prot = get_protocol()

    begin = tf.constant(begin.attr["value"].tensor)
    end = tf.constant(end.attr["value"].tensor)
    strides = tf.constant(strides.attr["value"].tensor)

    return prot.strided_slice(input, begin, end, strides=strides,
                              begin_mask=begin_mask, end_mask=end_mask,
                              ellipsis_mask=ellipsis_mask,
                              new_axis_mask=new_axis_mask,
                              shrink_axis_mask=shrink_axis_mask)


def pack(node: Any, inputs: List[str], output_lookup: Dict[str, Any]) -> Any:
    raise NotImplementedError()
    input1 = output_lookup[inputs[0]]
    input2 = output_lookup[inputs[1]]

    prot = get_protocol()

    return prot.stack([input1, input2], axis=node.attr["axis"].i)


def bias_add(node: Any, inputs: List[str], output_lookup: Dict[str, Any]) -> Any:
    raise NotImplementedError()

    input = output_lookup[inputs[0]]
    bias = output_lookup[inputs[1]]

    return input + bias


def maxpool(node: Any, inputs: List[str], output_lookup: Dict[str, Any]) -> Any:
    raise NotImplementedError()

    input = output_lookup[inputs[0]]

    return tf.nn.max_pool(input, list(node.attr["ksize"].list.i),
                          list(node.attr["strides"].list.i),
                          node.attr["padding"].s)


def shape(node: Any, inputs: List[str], output_lookup: Dict[str, Any]) -> Any:
    input = output_lookup[inputs[0]]

    return input.shape


def reshape(node: Any, inputs: List[str], output_lookup: Dict[str, Any]) -> Any:
    raise NotImplementedError()

    input = output_lookup[inputs[0]]
    shape = output_lookup[inputs[1]]

    return tf.reshape(input, shape)
