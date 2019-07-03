"""TFE Keras loss function"""
from tf_encrypted import get_protocol

class Loss(object):
  """Loss base class."""
  def __init__(self,
               fn,
               **kwargs):

    self.fn = fn
    self._fn_kwargs = kwargs

  def call(self, y_true, y_pred):
    """Invokes the `LossFunctionWrapper` instance.
    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.
    Returns:
      Loss values per sample.
    """
    return self.fn(y_true, y_pred, **self._fn_kwargs)

  def __call__(self, y_true, y_pred):
    """Invokes the `Loss` instance.

    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.
    """
    losses = self.call(y_true, y_pred)
    return losses


class BinaryCrossentropy(Loss):
  """Computes the cross-entropy loss between true l
  abels and predicted labels.
  """
  def __init__(self):
    super(BinaryCrossentropy, self).__init__(
        binary_crossentropy)

def binary_crossentropy(y_true, y_pred):

  batch_size = y_true.shape.as_list()[0]
  batch_size_inv = 1 / batch_size
  out = y_true * get_protocol().log(y_pred)
  out += (y_true.negative() + 1) * get_protocol().log(y_pred.negative() + 1)
  out = out.negative()
  bce = out.reduce_sum(axis=0) * batch_size_inv
  return bce
