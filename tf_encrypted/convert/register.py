"""Registry for the TF Encrypted Converter."""
import array
import logging
import os
from typing import Any, List
from collections import OrderedDict

import yaml
import numpy as np
import tensorflow as tf

from ..layers import Conv2D, Relu, Sigmoid, Dense, AveragePooling2D, MaxPooling2D
from ..keras.layers import BatchNormalization
from ..protocol.pond import PondPrivateTensor, PondMaskedTensor


def registry():
  """Map reserved names and scopes to their conversion functions."""
  reg = {
      'Placeholder': _placeholder,
      'Const': _constant,
      'Conv2D': _conv2d,
      'Relu': _relu,
      'Sigmoid': _sigmoid,
      'MatMul': _matmul,
      'Shape': _shape,
      'StridedSlice': _strided_slice,
      'Add': _add,
      'Sub': _sub,
      'Transpose': _transpose,
      'Reshape': _reshape,
      'Pack': _pack,
      'Rsqrt': _rsqrt,
      'Mul': _mul,
      'ExpandDims': _expand_dims,
      'AvgPool': _avgpool,
      'Squeeze': _squeeze,
      'ConcatV2': _concat,
      'BiasAdd': _bias_add,
      'MaxPool': _maxpool,
      'Pad': _pad,
      'BatchToSpaceND': _batch_to_space_nd,
      'SpaceToBatchND': _space_to_batch_nd,
      'ArgMax': _argmax,
      'required_space_to_batch_paddings': _required_space_to_batch_paddings,
      'flatten': _flatten,
      'conv2d': _keras_conv2d,
      'Slice': _slice,
      'Neg': _negative,
      'Split': _split,
      'SplitV': _split,
      'Identity': _identity,
      "GatherV2": _gather,
      "dense": _keras_dense,
      "batch_normalization_v1": _keras_batchnorm,
  }

  return reg


convert_dir = os.path.dirname(os.path.abspath(__file__))
specops_path = os.path.join(convert_dir, "specops.yaml")
with open(specops_path, "r") as stream:
  loaded_yaml = yaml.load(stream, Loader=yaml.SafeLoader)
  sorted_yaml = sorted(loaded_yaml.items(), key=lambda kv: kv[0])
  REGISTERED_SPECOPS = OrderedDict(sorted_yaml)


# pylint: disable=unused-argument
# pylint: disable=missing-docstring
def _placeholder(converter, node: Any, inputs: List[str]) -> Any:
  return tf.placeholder(node.attr["dtype"].type,
                        shape=node.attr["shape"].shape)


def _constant(converter, node: Any, inputs: List[str]) -> Any:
  # need to able to access the underlying weights return the node
  return node


def _identity(converter, node: Any, inputs: List[str]) -> Any:
  # need to able to access the underlying weights return the node
  return converter.outputs[inputs[0]]


def _matmul(converter, node: Any, inputs: List[str]) -> Any:
  a = converter.outputs[inputs[0]]
  b = converter.outputs[inputs[1]]

  tensor = b.attr["value"].tensor

  b_shape = [i.size for i in tensor.tensor_shape.dim]

  transpose_a = node.attr["transpose_a"].b
  transpose_b = node.attr["transpose_b"].b

  layer = Dense(a.shape.as_list(),
                b_shape[1],
                transpose_input=transpose_a,
                transpose_weight=transpose_b)

  dtype = tensor.dtype

  if dtype == tf.float32:
    nums = array.array('f', tensor.tensor_content)
  elif dtype == tf.float64:
    nums = array.array('d', tensor.tensor_content)
  else:
    raise TypeError("Unsupported dtype for weights")

  def inputter_fn():
    return tf.constant(np.array(nums).reshape(b_shape))

  w = converter.protocol.define_private_input(converter.model_provider,
                                              inputter_fn)

  layer.initialize(initial_weights=w)

  return layer.forward(a)


def _conv2d(converter, node, inputs):
  x_in = converter.outputs[inputs[0]]
  kernel = converter.outputs[inputs[1]]

  if isinstance(kernel, tf.NodeDef):
    shape = [i.size for i in kernel.attr["value"].tensor.tensor_shape.dim]
    w = _nodef_to_private_pond(converter, kernel)
  else:
    shape = kernel.shape.as_list()
    w = kernel

  fmt = node.attr["data_format"].s.decode('ascii')

  layer = Conv2D(x_in.shape.as_list(),
                 shape,
                 strides=int(max(node.attr["strides"].list.i)),
                 padding=node.attr["padding"].s.decode('ascii'),
                 channels_first=fmt == "NCHW")

  layer.initialize(initial_weights=w)

  out = layer.forward(x_in)

  return out


def _keras_conv2d(converter, interiors, inputs):
  x_in = converter.outputs[inputs[0]]

  conv_op = interiors["Conv2D"]
  kernel = interiors["kernel"]
  k = _nodef_to_private_pond(converter, kernel)
  try:
    bias = interiors["bias"]
    b = _nodef_to_private_pond(converter, bias)
    for ax in [0, -1, -1]:
      b = b.expand_dims(axis=ax)
  except KeyError:
    b = None

  input_shape = x_in.shape.as_list()
  shape = [i.size for i in kernel.attr["value"].tensor.tensor_shape.dim]
  fmt = conv_op.attr["data_format"].s.decode('ascii')
  strides = int(max(conv_op.attr["strides"].list.i))
  padding = conv_op.attr["padding"].s.decode('ascii')

  layer = Conv2D(
      input_shape, shape,
      strides=strides,
      padding=padding,
      channels_first=fmt == "NCHW"
  )

  layer.initialize(initial_weights=k, initial_bias=b)
  out = layer.forward(x_in)

  return out


def _keras_dense(converter, interiors, inputs):
  x_in = converter.outputs[inputs[0]]

  kernel = interiors["kernel"]
  k = _nodef_to_private_pond(converter, kernel)
  try:
    bias = interiors["bias"]
    b = _nodef_to_private_pond(converter, bias)
  except KeyError:
    b = None

  input_shape = x_in.shape.as_list()
  shape = [i.size for i in kernel.attr["value"].tensor.tensor_shape.dim]

  layer = Dense(input_shape,
                out_features=shape[1])

  layer.initialize(initial_weights=k, initial_bias=b)
  out = layer.forward(x_in)

  return out


def _keras_batchnorm(converter, interiors, inputs):
  x_in = converter.outputs[inputs[0]]

  bn_op = interiors["FusedBatchNorm"]
  fmt = bn_op.attr["data_format"].s.decode('ascii')

  gamma = _nodef_to_numpy_array(interiors["gamma"])
  gamma_init = tf.keras.initializers.Constant(gamma)

  beta = _nodef_to_numpy_array(interiors["beta"])
  beta_init = tf.keras.initializers.Constant(beta)

  moving_mean = _nodef_to_numpy_array(interiors["moving_mean"])
  moving_mean_init = tf.keras.initializers.Constant(moving_mean)

  moving_variance = _nodef_to_numpy_array(interiors["moving_variance"])
  moving_variance_init = tf.keras.initializers.Constant(moving_variance)

  input_shape = x_in.shape.as_list()

  layer = BatchNormalization(input_shape=input_shape,
                             axis=(3 if fmt == "NHWC" else 1),
                             gamma_initializer=gamma_init,
                             beta_initializer=beta_init,
                             moving_mean_initializer=moving_mean_init,
                             moving_variance_initializer=moving_variance_init)

  return layer(x_in)


def _relu(converter, node: Any, inputs: List[str]) -> Any:
  x_in = converter.outputs[inputs[0]]

  return Relu(x_in.shape.as_list()).forward(x_in)


def _sigmoid(converter, node: Any, inputs: List[str]) -> Any:
  x_in = converter.outputs[inputs[0]]

  return Sigmoid(x_in.shape.as_list()).forward(x_in)


def _strided_slice(converter, node: Any, inputs: List[str]) -> Any:
  x_in = converter.outputs[inputs[0]]

  if isinstance(x_in, tf.NodeDef):
    input_out = _nodef_to_private_pond(converter, x_in)
  else:
    input_out = x_in

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

  return converter.protocol.strided_slice(input_out, begin, end,
                                          strides=strides,
                                          begin_mask=begin_mask,
                                          end_mask=end_mask,
                                          ellipsis_mask=ellipsis_mask,
                                          new_axis_mask=new_axis_mask,
                                          shrink_axis_mask=shrink_axis_mask)


def _pack(converter, node: Any, inputs: List[str]) -> Any:
  final_inputs = []

  for x_in in inputs:
    input_c = converter.outputs[x_in]
    if isinstance(input_c, tf.NodeDef):
      final_inputs.append(_nodef_to_private_pond(converter, input_c))
    else:
      final_inputs.append(input_c)

  return converter.protocol.stack(final_inputs, axis=node.attr["axis"].i)


def _bias_add(converter, node: Any, inputs: List[str]) -> Any:
  a = converter.outputs[inputs[0]]
  b = converter.outputs[inputs[1]]

  if isinstance(a, tf.NodeDef):
    a_out = _nodef_to_private_pond(converter, a)
  else:
    a_out = a

  if isinstance(b, tf.NodeDef):
    b_out = _nodef_to_private_pond(converter, b)
  else:
    b_out = b

  return converter.protocol.add(a_out, b_out)


def _maxpool(converter, node: Any, inputs: List[str]) -> Any:
  x_in = converter.outputs[inputs[0]]

  ksize = node.attr["ksize"].list.i
  s = node.attr["strides"].list.i

  padding = node.attr["padding"].s.decode('ascii')
  pool_size = [ksize[1], ksize[2]]
  strides = [s[1], s[2]]

  shape = [int(i) for i in x_in.shape]

  channels_first = node.attr["data_format"].s.decode('ascii') == "NCHW"

  pooler = MaxPooling2D(shape, pool_size, strides, padding, channels_first)

  out = pooler.forward(x_in)

  return out


def _shape(converter, node: Any, inputs: List[str]) -> Any:
  x_in = converter.outputs[inputs[0]]

  return x_in.shape


def _reshape(converter, node: Any, inputs: List[str]) -> Any:
  x_in = converter.outputs[inputs[0]]
  shape = converter.outputs[inputs[1]]

  tensor = shape.attr["value"].tensor
  dtype = shape.attr["dtype"].type
  if dtype == tf.int32:
    nums = array.array('i', tensor.tensor_content)
  elif dtype == tf.int64:
    nums = array.array('l', tensor.tensor_content)
  else:
    raise TypeError("Unsupported dtype for reshape shape")

  return converter.protocol.reshape(x_in, list(nums))


def _transpose(converter, node: Any, inputs: List[str]) -> Any:
  x_in = converter.outputs[inputs[0]]
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

  return converter.protocol.transpose(x_in, np.array(nums).reshape(shape))


def _expand_dims(converter, node: Any, inputs: List[str]) -> Any:
  x_in = converter.outputs[inputs[0]]

  if isinstance(x_in, tf.NodeDef):
    input_out = _nodef_to_private_pond(converter, x_in)
  else:
    input_out = x_in

  input_axis = converter.outputs[inputs[1]]
  axis_attr = input_axis.attr["value"].tensor.int_val
  axis_val = array.array('i', axis_attr)[0]

  return converter.protocol.expand_dims(input_out, axis_val)


def _negative(converter, node: Any, inputs: List[str]) -> Any:
  x_in = converter.outputs[inputs[0]]

  if isinstance(x_in, tf.NodeDef):
    input_out = _nodef_to_private_pond(converter, x_in)
  else:
    input_out = x_in

  return converter.protocol.negative(input_out)


def _gather(converter, node: Any, inputs: List[str]) -> Any:
  x_in = converter.outputs[inputs[0]]
  indices = converter.outputs[inputs[1]]
  axis = converter.outputs[inputs[2]]

  if isinstance(x_in, tf.NodeDef):
    input_out = _nodef_to_private_pond(converter, x_in)
  else:
    input_out = x_in

  indices_out = list(_nodef_to_numpy_array(indices))

  axis_val = axis.attr["value"].tensor.int_val[0]

  return converter.protocol.gather(input_out, indices_out, axis_val)


def _squeeze(converter, node: Any, inputs: List[str]) -> Any:
  x_in = converter.outputs[inputs[0]]

  axis = node.attr["squeeze_dims"].list.i

  return converter.protocol.squeeze(x_in, list(axis))


def _split(converter, node: Any, inputs: List[str]) -> Any:
  if node.op == "SplitV":
    #node.op is SplitV when num_or_size_splits is a list
    x_in = converter.outputs[inputs[0]]
    size_splits = converter.outputs[inputs[1]]
    axis = converter.outputs[inputs[2]]

    size_splits = size_splits.attr["value"].tensor
    num_or_size_splits = list(array.array('I', size_splits.tensor_content))

  else:
    #node.op is Split when num_or_size_splits is an integer
    axis = converter.outputs[inputs[0]]
    x_in = converter.outputs[inputs[1]]

    num_or_size_splits = node.attr["num_split"].i

  if isinstance(x_in, tf.NodeDef):
    input_out = _nodef_to_private_pond(converter, x_in)
  else:
    input_out = x_in

  axis_val = axis.attr["value"].tensor.int_val[0]

  return converter.protocol.split(input_out, num_or_size_splits, axis_val)


def _pad(converter, node: Any, inputs: List[str]) -> Any:
  x_in = converter.outputs[inputs[0]]
  p = (converter.outputs[inputs[1]])

  paddings_t = p.attr["value"].tensor

  paddings_arr = list(array.array('I', paddings_t.tensor_content))
  paddings_lst = [paddings_arr[i:i + 2]
                  for i in range(0, len(paddings_arr), 2)]

  return converter.protocol.pad(x_in, paddings_lst)


def _rsqrt(converter, node: Any, inputs: List[str]) -> Any:
  x_in = converter.outputs[inputs[0]]

  if isinstance(x_in, tf.NodeDef):
    tensor = x_in.attr["value"].tensor
    shape = [i.size for i in tensor.tensor_shape.dim]

    dtype = x_in.attr["dtype"].type
    if dtype == tf.float32:
      nums = array.array('f', tensor.tensor_content)
    elif dtype == tf.float64:
      nums = array.array('d', tensor.tensor_content)

    else:
      raise TypeError("Unsupported dtype for rsqrt")

    def inputter_fn():
      return tf.constant(1 / np.sqrt(np.array(nums).reshape(shape)))

  else:
    # XXX this is a little weird but the input into rsqrt is public and
    # being used only for batchnorm at the moment
    decoded = converter.protocol._decode(x_in.value_on_0, True)  # pylint: disable=protected-access

    def inputter_fn():
      return tf.rsqrt(decoded)

  x = converter.protocol.define_public_input(
      converter.model_provider, inputter_fn)

  return x


def _add(converter, node: Any, inputs: List[str]) -> Any:
  a = converter.outputs[inputs[0]]
  b = converter.outputs[inputs[1]]

  if isinstance(a, tf.NodeDef):
    a_out = _nodef_to_public_pond(converter, a)
  else:
    a_out = a

  if isinstance(b, tf.NodeDef):
    b_out = _nodef_to_public_pond(converter, b)
  else:
    b_out = b

  return converter.protocol.add(a_out, b_out)


def _sub(converter, node: Any, inputs: List[str]) -> Any:
  a = converter.outputs[inputs[0]]
  b = converter.outputs[inputs[1]]

  if isinstance(a, tf.NodeDef):
    a_out = _nodef_to_public_pond(converter, a)
  else:
    a_out = a

  if isinstance(b, tf.NodeDef):
    b_out = _nodef_to_public_pond(converter, b)
  else:
    b_out = b

  return converter.protocol.sub(a_out, b_out)


def _mul(converter, node: Any, inputs: List[str]) -> Any:
  a = converter.outputs[inputs[0]]
  b = converter.outputs[inputs[1]]

  if isinstance(a, tf.NodeDef):
    a_out = _nodef_to_public_pond(converter, a)
  else:
    a_out = a

  if isinstance(b, tf.NodeDef):
    b_out = _nodef_to_public_pond(converter, b)
  else:
    b_out = b

  return converter.protocol.mul(a_out, b_out)


def _avgpool(converter, node: Any, inputs: List[str]) -> Any:
  x_in = converter.outputs[inputs[0]]

  ksize = node.attr["ksize"].list.i
  s = node.attr["strides"].list.i

  padding = node.attr["padding"].s.decode('ascii')
  pool_size = [ksize[1], ksize[2]]
  strides = [s[1], s[2]]

  shape = [int(i) for i in x_in.shape]

  channels_first = node.attr["data_format"].s.decode('ascii') == "NCHW"

  avg = AveragePooling2D(shape, pool_size, strides, padding, channels_first)

  out = avg.forward(x_in)

  return out


def _concat(converter, node: Any, inputs: List[str]) -> Any:
  input_list = [converter.outputs[inputs[i]] for i in range(len(inputs) - 1)]
  axis = converter.outputs[inputs[-1]]
  axis_int = axis.attr["value"].tensor.int_val[0]

  return converter.protocol.concat(input_list, axis_int)


def _batch_to_space_nd(converter, node, inputs):
  x_in = converter.outputs[inputs[0]]
  block_shape = converter.outputs[inputs[1]].attr["value"].tensor
  crops = converter.outputs[inputs[2]].attr["value"].tensor

  return converter.protocol.batch_to_space_nd(x_in, block_shape, crops)


def _space_to_batch_nd(converter, node, inputs):
  x_in = converter.outputs[inputs[0]]
  block_shape = converter.outputs[inputs[1]].attr["value"].tensor
  paddings = converter.outputs[inputs[2]].attr["value"].tensor

  return converter.protocol.space_to_batch_nd(x_in, block_shape, paddings)


def _flatten(converter, node, inputs):
  x_in = converter.outputs[inputs[0]]

  shape = x_in.shape.as_list()
  non_batch = 1
  for dim in shape[1:]:
    non_batch *= dim

  return converter.protocol.reshape(x_in, [-1, non_batch])


def _required_space_to_batch_paddings(converter, node, inputs: List[str]):

  inputs_node = [converter.outputs[inputs[i]] for i in range(len(inputs))]
  inputs_int32 = []
  for x_in in inputs_node:
    pvt_check = isinstance(x_in, PondPrivateTensor)
    msk_check = isinstance(x_in, PondMaskedTensor)
    if pvt_check or msk_check:
      logging.warning(("Revealing private input: "
                       "required_space_to_batch_paddings assumes public "
                       "input."))
      inputs_int32.append(tf.cast(x_in.reveal().decode(), tf.int32))
    elif isinstance(x_in, tf.NodeDef):
      inputs_int32.append(_nodef_to_numpy_array(x_in))
    else:
      raise TypeError("Unexpected input of type {}.".format(type(x_in)))

  if len(inputs_int32) == 2:
    input_shape, block_shape = inputs_int32

    def inputter_pad():
      pads, _ = tf.required_space_to_batch_paddings(input_shape, block_shape)
      return tf.cast(pads, tf.float64)

    def inputter_crop():
      _, crops = tf.required_space_to_batch_paddings(input_shape, block_shape)
      return tf.cast(crops, tf.float64)
  else:
    base_paddings, input_shape, block_shape = inputs_int32

    def inputter_pad():
      pads, _ = tf.required_space_to_batch_paddings(
          input_shape,
          block_shape,
          base_paddings=base_paddings,
      )
      return tf.cast(pads, tf.float64)

    def inputter_crop():
      _, crops = tf.required_space_to_batch_paddings(
          input_shape,
          block_shape,
          base_paddings=base_paddings,
      )
      return tf.cast(crops, tf.float64)

  pad_private = converter.protocol.define_public_input(
      converter.model_provider, inputter_pad)
  crop_private = converter.protocol.define_public_input(
      converter.model_provider, inputter_crop)

  return (pad_private, crop_private)


def _argmax(converter, node, inputs):
  x_in = converter.outputs[inputs[0]]
  axis = converter.outputs[inputs[1]].attr["value"].tensor.int_val[0]

  return converter.protocol.argmax(x_in, axis=axis)


def _slice(converter, node, inputs):
  x_in = converter.outputs[inputs[0]]
  begin = _nodef_to_numpy_array(converter.outputs[inputs[1]])
  size = _nodef_to_numpy_array(converter.outputs[inputs[2]])

  if isinstance(x_in, tf.NodeDef):
    input_out = _nodef_to_private_pond(converter, x_in)
  else:
    input_out = x_in

  # Slice is a special case of strided_slice. Slice takes size (the number of
  # elements we want to slice) as an input. However strided_slice takes end
  # (integer until which the slicing takes place) as input.
  # We can infere the end parameter with : end[i] = begin[i] + size[i].
  # If size is negative, the stepping go towards smaller indices.
  # In this case we can infer the end parameter with: end[i] = input_shape[i] - size[i] + 1
  end = np.zeros(len(begin))
  input_shape = x_in.shape.as_list()

  # if size is negative take the input dimension
  for i in range(len(end)):  # pylint: disable=consider-using-enumerate
    if size[i] < 0:
      end[i] = input_shape[i] - size[i] + 1
    else:
      end[i] = begin[i] + size[i]

  return converter.protocol.strided_slice(input_out, begin, end)


# pylint: enable=unused-argument
# pylint: enable=missing-docstring
def _nodef_to_public_pond(converter, x):
  """Map a NodeDef x to a PublicPondTensor."""
  dtype = x.attr["dtype"].type
  x_shape = [i.size for i in x.attr["value"].tensor.tensor_shape.dim]

  if not x_shape:
    if dtype == tf.float32:
      nums = x.attr["value"].tensor.float_val
    elif dtype == tf.float64:
      nums = x.attr["value"].tensor.float_val
    elif dtype == tf.int32:
      nums = x.attr["value"].tensor.int_val
    else:
      raise TypeError("Unsupported dtype")

    def inputter_fn():
      return tf.constant(np.array(nums).reshape(1, 1))

  else:
    if dtype == tf.float32:
      nums = array.array('f', x.attr["value"].tensor.tensor_content)
    elif dtype == tf.float64:
      nums = array.array('d', x.attr["value"].tensor.tensor_content)
    elif dtype == tf.int32:
      nums = array.array('i', x.attr["value"].tensor.tensor_content)
    else:
      raise TypeError("Unsupported dtype")

    def inputter_fn():
      return tf.constant(np.array(nums).reshape(x_shape))

  x_public = converter.protocol.define_public_input(
      converter.model_provider, inputter_fn)

  return x_public


def _nodef_to_private_pond(converter, x):
  """Map a NodeDef x to a PrivatePondTensor."""
  dtype = x.attr["dtype"].type
  warn_msg = "Unexpected dtype {} found at node {}"
  err_msg = "Unsupported dtype {} found at node {}"

  x_shape = [i.size for i in x.attr["value"].tensor.tensor_shape.dim]

  if not x_shape:
    if dtype == tf.float32:
      nums = x.attr["value"].tensor.float_val
    elif dtype == tf.float64:
      nums = x.attr["value"].tensor.float_val
    elif dtype == tf.int32:
      logging.warning(warn_msg, dtype, x.name)
      nums = x.attr["value"].tensor.int_val
    else:
      raise TypeError(err_msg.format(dtype, x.name))

    def inputter_fn():
      return tf.constant(np.array(nums).reshape(1, 1))

  else:
    if dtype == tf.float32:
      nums = array.array('f', x.attr["value"].tensor.tensor_content)
    elif dtype == tf.float64:
      nums = array.array('d', x.attr["value"].tensor.tensor_content)
    elif dtype == tf.int32:
      logging.warning(warn_msg, dtype, x.name)
      nums = array.array('i', x.attr["value"].tensor.tensor_content)
    else:
      raise TypeError(err_msg.format(dtype, x.name))

    def inputter_fn():
      return tf.constant(np.array(nums).reshape(x_shape))

  x_private = converter.protocol.define_private_input(
      converter.model_provider, inputter_fn)

  return x_private


def _nodef_to_numpy_array(x):
  """Map a NodeDef x to a np.array."""
  dtype = x.attr["dtype"].type
  x_shape = [i.size for i in x.attr["value"].tensor.tensor_shape.dim]

  content = x.attr["value"].tensor.tensor_content

  if dtype == tf.float32:
    type_code = 'f'
    if not content:
      content = x.attr["value"].tensor.float_val
  elif dtype == tf.float64:
    type_code = 'd'
    if not content:
      content = x.attr["value"].tensor.double_val
  elif dtype == tf.int32:
    type_code = 'i'
    if not content:
      content = x.attr["value"].tensor.int_val
  else:
    raise TypeError("Unsupported dtype")

  nums = array.array(type_code, content)

  return np.array(nums).reshape(x_shape)
