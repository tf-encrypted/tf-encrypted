import tensorflow as tf
import numpy as np
import array
from typing import Any, Dict, List

from ..layers import Conv2D, Relu, Sigmoid, Dense, AveragePooling2D
from .convert import Converter, ConvertInputProvider

from tensorflow_encrypted.protocol.pond import PondPublicTensor


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
        'Add': add,
        'Sub': sub,
        'Transpose': transpose,
        'Reshape': reshape,
        'Pack': pack,
        'Rsqrt': rsqrt,
        'Mul': mul,
        'ExpandDims': expand_dims,
        'AvgPool': avgpool,
        'Squeeze': squeeze
        # 'Pack': pack,
        # 'BiasAdd': bias_add,
        # 'MaxPool': maxpool,
    }

    return reg


def placeholder(converter: Converter, node: Any, inputs: List[str]) -> Any:
    return tf.placeholder(node.attr["dtype"].type,
                          shape=node.attr["shape"].shape)


def constant(converter: Converter, node: Any, inputs: List[str]) -> Any:
    # need to able to access the underlying weights return the node
    return node


def matmul(converter: Converter, node: Any, inputs: List[str]) -> Any:
    a = converter.outputs[inputs[0]]
    b = converter.outputs[inputs[1]]

    tensor = b.attr["value"].tensor

    b_shape = [i.size for i in tensor.tensor_shape.dim]

    layer = Dense(a.shape.as_list(), b_shape[1])

    dtype = tensor.dtype

    if dtype == tf.float32:
        nums = array.array('f', tensor.tensor_content)
    elif dtype == tf.float64:
        nums = array.array('d', tensor.tensor_content)
    else:
        raise TypeError("Unsupported dtype for weights")

    provider = ConvertInputProvider(converter.weights_provider,
                                    np.array(nums).reshape(b_shape))
    w = converter.protocol.define_private_input(provider)

    layer.initialize(initial_weights=w)

    return layer.forward(a)


def conv2d(converter: Converter, node: Any, inputs: List[str]) -> Any:
    input = converter.outputs[inputs[0]]
    filter = converter.outputs[inputs[1]]

    shape = [i.size for i in filter.attr["value"].tensor.tensor_shape.dim]
    dtype = filter.attr["dtype"].type
    format = node.attr["data_format"].s.decode('ascii')

    layer = Conv2D(
        input.shape.as_list(), shape,
        strides=int(node.attr["strides"].list.i[0]),
        padding=node.attr["padding"].s.decode('ascii'),
        channels_first=format == "NCHW"
    )

    if dtype == tf.float32:
        nums = array.array('f', filter.attr["value"].tensor.tensor_content)
    elif dtype == tf.float64:
        nums = array.array('d', filter.attr["value"].tensor.tensor_content)
    else:
        raise TypeError("Unsupported dtype for weights")

    provider = ConvertInputProvider(converter.weights_provider,
                                    np.array(nums).reshape(shape))
    w = converter.protocol.define_private_input(provider)

    layer.initialize(initial_weights=w)

    out = layer.forward(input)

    return out


def relu(converter: Converter, node: Any, inputs: List[str]) -> Any:
    input = converter.outputs[inputs[0]]

    return Relu(input.shape.as_list()).forward(input)


def sigmoid(converter: Converter, node: Any, inputs: List[str]) -> Any:
    input = converter.outputs[inputs[0]]

    return Sigmoid(input.shape.as_list()).forward(input)


def strided_slice(converter: Converter, node: Any, inputs: List[str]) -> Any:
    input = converter.outputs[inputs[0]]
    begin = converter.outputs[inputs[1]]
    end = converter.outputs[inputs[2]]
    strides = converter.outputs[inputs[3]]

    begin_mask = node.attr["begin_mask"].i
    end_mask = node.attr["end_mask"].i
    ellipsis_mask = node.attr["ellipsis_mask"].i
    new_axis_mask = node.attr["new_axis_mask"].i
    shrink_axis_mask = node.attr["shrink_axis_mask"].i

    begin = tf.constant(begin.attr["value"].tensor)
    end = tf.constant(end.attr["value"].tensor)
    strides = tf.constant(strides.attr["value"].tensor)

    return converter.protocol.strided_slice(input, begin, end, strides=strides,
                                            begin_mask=begin_mask,
                                            end_mask=end_mask,
                                            ellipsis_mask=ellipsis_mask,
                                            new_axis_mask=new_axis_mask,
                                            shrink_axis_mask=shrink_axis_mask)


def pack(converter: Converter, node: Any, inputs: List[str]) -> Any:
    final_inputs = []

    for input in inputs:
        final_inputs.append(converter.outputs[input])

    return converter.protocol.stack(final_inputs, axis=node.attr["axis"].i)


def bias_add(converter: Converter, node: Any, inputs: List[str]) -> Any:
    raise NotImplementedError()

    input = converter.outputs[inputs[0]]
    bias = converter.outputs[inputs[1]]

    return input + bias


def maxpool(converter: Converter, node: Any, inputs: List[str]) -> Any:
    raise NotImplementedError()

    input = converter.outputs[inputs[0]]

    return tf.nn.max_pool(input, list(node.attr["ksize"].list.i),
                          list(node.attr["strides"].list.i),
                          node.attr["padding"].s)


def shape(converter: Converter, node: Any, inputs: List[str]) -> Any:
    input = converter.outputs[inputs[0]]

    return input.shape


def reshape(converter: Converter, node: Any, inputs: List[str]) -> Any:
    input = converter.outputs[inputs[0]]
    shape = converter.outputs[inputs[1]]

    tensor = shape.attr["value"].tensor
    dtype = shape.attr["dtype"].type
    if dtype == tf.int32:
        nums = array.array('i', tensor.tensor_content)
    elif dtype == tf.int64:
        nums = array.array('l', tensor.tensor_content)
    else:
        raise TypeError("Unsupported dtype for reshape shape")

    return converter.protocol.reshape(input, list(nums))


def transpose(converter: Converter, node: Any, inputs: List[str]) -> Any:
    input = converter.outputs[inputs[0]]
    perm = converter.outputs[inputs[1]]

    tensor = perm.attr["value"].tensor
    shape = [i.size for i in tensor.tensor_shape.dim]

    dtype = perm.attr["dtype"].type
    if dtype == tf.int32:
        nums = array.array('i', tensor.tensor_content)
    elif dtype == tf.int64:
        nums = array.array('l', tensor.tensor_content)
    else:
        raise TypeError("Unsupported dtype for transpose perm")

    return converter.protocol.transpose(input, np.array(nums).reshape(shape))


def expand_dims(converter: Converter, node: Any, inputs: List[str]) -> Any:
    input = converter.outputs[inputs[0]]
    # axis = converter.outputs[inputs[1]]

    axis_val = node.attr["axis"].i

    return converter.protocol.expand_dims(input, axis_val)


def squeeze(converter: Converter, node: Any, inputs: List[str]) -> Any:
    input = converter.outputs[inputs[0]]
    axis = node.attr["squeeze_dims"].list.i

    return converter.protocol.squeeze(input, list(axis))


def rsqrt(converter: Converter, node: Any, inputs: List[str]) -> Any:
    input = converter.outputs[inputs[0]]

    tensor = input.attr["value"].tensor
    shape = [i.size for i in tensor.tensor_shape.dim]

    dtype = input.attr["dtype"].type
    if dtype == tf.float32:
        nums = array.array('f', tensor.tensor_content)
    elif dtype == tf.float64:
        nums = array.array('d', tensor.tensor_content)
    else:
        raise TypeError("Unsupported dtype for rsqrt")

    x = 1 / np.sqrt(np.array(nums).reshape(shape))

    provider = ConvertInputProvider(converter.weights_provider, x)

    x = converter.protocol.define_public_input(provider)

    return x


def add(converter: Converter, node: Any, inputs: List[str]) -> Any:
    a = converter.outputs[inputs[0]]
    b = converter.outputs[inputs[1]]

    if isinstance(a, tf.NodeDef):
        a_out = nodef_to_public_pond(converter, a)
    else:
        a_out = a

    if isinstance(b, tf.NodeDef):
        b_out = nodef_to_public_pond(converter, b)
    else:
        b_out = b

    return converter.protocol.add(a_out, b_out)


def sub(converter: Converter, node: Any, inputs: List[str]) -> Any:
    a = converter.outputs[inputs[0]]
    b = converter.outputs[inputs[1]]

    if isinstance(a, tf.NodeDef):
        a_out = nodef_to_public_pond(converter, a)
    else:
        a_out = a

    if isinstance(b, tf.NodeDef):
        b_out = nodef_to_public_pond(converter, b)
    else:
        b_out = b

    return converter.protocol.sub(a_out, b_out)


def mul(converter: Converter, node: Any, inputs: List[str]) -> Any:
    a = converter.outputs[inputs[0]]
    b = converter.outputs[inputs[1]]

    if isinstance(a, tf.NodeDef):
        a_out = nodef_to_public_pond(converter, a)
    else:
        a_out = a

    if isinstance(b, tf.NodeDef):
        b_out = nodef_to_public_pond(converter, b)
    else:
        b_out = b

    return converter.protocol.mul(a_out, b_out)


def avgpool(converter: Converter, node: Any, inputs: List[str]) -> Any:
    input = converter.outputs[inputs[0]]

    ksize = node.attr["ksize"].list.i
    s = node.attr["strides"].list.i

    padding = node.attr["padding"].s.decode('ascii')
    pool_size = [ksize[1], ksize[2]]
    strides = [s[1], s[2]]

    shape = [int(i) for i in input.shape]

    channels_first = node.attr["data_format"].s.decode('ascii') == "NCHW"

    avg = AveragePooling2D(shape, pool_size, strides, padding, channels_first)

    out = avg.forward(input)

    return out


def nodef_to_public_pond(converter: Converter, x: Any) -> PondPublicTensor:
    dtype = x.attr["dtype"].type
    x_shape = [i.size for i in x.attr["value"].tensor.tensor_shape.dim]

    if dtype == tf.float32:
        nums = array.array('f', x.attr["value"].tensor.tensor_content)
    elif dtype == tf.float64:
        nums = array.array('d', x.attr["value"].tensor.tensor_content)
    else:
        raise TypeError("Unsupported dtype")

    provider = ConvertInputProvider(converter.weights_provider,
                                    np.array(nums).reshape(x_shape))

    x_public = converter.protocol.define_public_input(provider)

    return x_public
