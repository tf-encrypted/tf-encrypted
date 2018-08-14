import tensorflow as tf

from typing import Any, Dict, List


def register() -> Dict[str, Any]:
    reg = {}

    reg['Placeholder'] = placeholder
    reg['Const'] = constant
    reg['Conv2D'] = conv2d
    reg['Relu'] = relu
    reg['BiasAdd'] = bias_add
    reg['MaxPool'] = maxpool
    reg['Shape'] = shape
    reg['StridedSlice'] = strided_slice
    reg['Pack'] = pack
    reg['Reshape'] = reshape
    reg['MatMul'] = matmul
    reg['Softmax'] = softmax

    return reg


def placeholder(node: Any, inputs: List[str], output_lookup: Dict[str, Any]) -> Any:
    return tf.placeholder(node.attr["dtype"].type,
                          shape=node.attr["shape"].shape)


def constant(node: Any, inputs: List[str], output_lookup: Dict[str, Any]) -> Any:
    return tf.constant(node.attr["value"].tensor)


def conv2d(node: Any, inputs: List[str], output_lookup: Dict[str, Any]) -> Any:
    input = output_lookup[inputs[0]]
    filter = output_lookup[inputs[1]]

    return tf.nn.conv2d(input, filter, list(node.attr["strides"].list.i),
                        node.attr["padding"].s)


def relu(node: Any, inputs: List[str], output_lookup: Dict[str, Any]) -> Any:
    input = output_lookup[inputs[0]]

    return tf.nn.relu(input)


def bias_add(node: Any, inputs: List[str], output_lookup: Dict[str, Any]) -> Any:
    input = output_lookup[inputs[0]]
    bias = output_lookup[inputs[1]]

    return input + bias


def maxpool(node: Any, inputs: List[str], output_lookup: Dict[str, Any]) -> Any:
    input = output_lookup[inputs[0]]

    return tf.nn.max_pool(input, list(node.attr["ksize"].list.i),
                          list(node.attr["strides"].list.i),
                          node.attr["padding"].s)


def shape(node: Any, inputs: List[str], output_lookup: Dict[str, Any]) -> Any:
    input = output_lookup[inputs[0]]

    return tf.shape(input, out_type=node.attr["out_type"].type)


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

    return tf.strided_slice(input, begin, end, strides=strides,
                            begin_mask=begin_mask, end_mask=end_mask,
                            ellipsis_mask=ellipsis_mask,
                            new_axis_mask=new_axis_mask,
                            shrink_axis_mask=shrink_axis_mask)


def pack(node: Any, inputs: List[str], output_lookup: Dict[str, Any]) -> Any:
    input1 = output_lookup[inputs[0]]
    input2 = output_lookup[inputs[1]]

    return tf.stack([input1, input2], axis=node.attr["axis"].i)


def reshape(node: Any, inputs: List[str], output_lookup: Dict[str, Any]) -> Any:
    input = output_lookup[inputs[0]]
    shape = output_lookup[inputs[1]]

    return tf.reshape(input, shape)


def matmul(node: Any, inputs: List[str], output_lookup: Dict[str, Any]) -> Any:
    a = output_lookup[inputs[0]]
    b = output_lookup[inputs[1]]

    return tf.matmul(a, b)  # consider transpose


def softmax(node: Any, inputs: List[str], output_lookup: Dict[str, Any]) -> Any:
    logits = output_lookup[inputs[0]]

    return tf.nn.softmax(logits)  # check axis?
