"""TFE Keras loss function"""
from tf_encrypted import get_protocol

class Loss():
  """Loss base class."""
  def __init__(self,
               loss_fn,
               **kwargs):

    self.loss_fn = loss_fn
    self._fn_kwargs = kwargs

  def call(self, y_true, y_pred):
    """Invokes the `LossFunctionWrapper` instance.
    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.
    Returns:
      Loss values per sample.
    """
    return self.loss_fn(y_true, y_pred, **self._fn_kwargs)

  def __call__(self, y_true, y_pred):
    """Invokes the `Loss` instance.

    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.
    """
    return self.call(y_true, y_pred)


class BinaryCrossentropy(Loss):
  """Computes the cross-entropy loss between true
  labels and predicted labels.
  """
  def __init__(self):
    super(BinaryCrossentropy, self).__init__(
        binary_crossentropy)

  def grad(self, y_true, y_pred):
    return y_pred - y_true

def binary_crossentropy(y_true, y_pred):

  batch_size = y_true.shape.as_list()[0]
  batch_size_inv = 1 / batch_size
  out = y_true * get_protocol().log(y_pred)
  out += (1 - y_true) * get_protocol().log(1 - y_pred)
  out = out.negative()
  bce = out.reduce_sum(axis=0) * batch_size_inv
  return bce


class MeanSquaredError(Loss):
  """Computes the MSE loss between true
  labels and predicted labels.
  """
  def __init__(self):
    super(MeanSquaredError, self).__init__(
        mean_squared_error)

  def grad(self, y_true, y_pred):
    batch_size = y_true.shape.as_list()[0] 
    batch_size_inv = 1 / batch_size  
    return 2 * (y_pred - y_true) * batch_size_inv 

def mean_squared_error(y_true, y_pred):
    batch_size = y_true.shape.as_list()[0] 
    batch_size_inv = 1 / batch_size  
    out = y_true - y_pred 
    out=out.square() 
    mse_loss = out.reduce_sum(axis=0) * batch_size_inv
    return  mse_loss