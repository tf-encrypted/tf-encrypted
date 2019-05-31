"""Core layers such as Reshape"""

from tf_encrypted.keras.engine import Layer

class Reshape(Layer):
  """Reshapes an output to a certain shape.
  Arguments:
    target_shape: Target shape. Tuple of integers,
      does not include the samples dimension (batch size).
  Input shape:
    Arbitrary, although all dimensions in the input shaped must be fixed.
    Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.
  Output shape:
    `(batch_size,) + target_shape`
  Example:
  ```python
  # as first layer in a Sequential model
  model = Sequential()
  model.add(Reshape((3, 4), input_shape=(12,)))
  # now: model.output_shape == (None, 3, 4)
  # note: `None` is the batch dimension
  # as intermediate layer in a Sequential model
  model.add(Reshape((6, 2)))
  # now: model.output_shape == (None, 6, 2)
  # also supports shape inference using `-1` as dimension
  model.add(Reshape((-1, 2, 2)))
  # now: model.output_shape == (None, 3, 2, 2)
  ```
  """

  def __init__(self, target_shape, **kwargs):
    super(Reshape, self).__init__(**kwargs)
    self.target_shape = tuple(target_shape)

  def _fix_unknown_dimension(self, input_shape, output_shape):
    """Find and replace a missing dimension in an output shape.
    This is a near direct port of the internal Numpy function
    `_fix_unknown_dimension` in `numpy/core/src/multiarray/shape.c`
    Arguments:
      input_shape: Shape of array being reshaped
      output_shape: Desired shape of the array with at most
        a single -1 which indicates a dimension that should be
        derived from the input shape.
    Returns:
      The new output shape with a -1 replaced with its computed value.
    Raises:
      ValueError: If the total array size of the output_shape is
      different than the input_shape, or more than one unknown dimension
      is specified.
    """
    output_shape = list(output_shape)
    msg = 'total size of new array must be unchanged'

    known, unknown = 1, None
    for index, dim in enumerate(output_shape):
      if dim < 0:
        if unknown is None:
          unknown = index
        else:
          raise ValueError('Can only specify one unknown dimension.')
      else:
        known *= dim

    original = np.prod(input_shape, dtype=int)
    if unknown is not None:
      if known == 0 or original % known != 0:
        raise ValueError(msg)
      output_shape[unknown] = original // known
    elif original != known:
      raise ValueError(msg)
    return output_shape

  def compute_output_shape(self, input_shape):
    if None in input_shape[1:]:
      output_shape = [input_shape[0]]
      # input shape (partially) unknown? replace -1's with None's
      output_shape += tuple(s if s != -1 else None for s in self.target_shape)
    else:
      output_shape = [input_shape[0]]
      output_shape += self._fix_unknown_dimension(input_shape[1:],
                                                  self.target_shape)
    return output_shape

  def call(self, inputs):
    return inputs.reshape((int(inputs.shape[0]),) + self.target_shape)
