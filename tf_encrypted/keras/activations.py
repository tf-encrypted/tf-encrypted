# pylint: disable=inconsistent-return-statements
"""Provide activation functions"""
import tf_encrypted as tfe


def relu(x):
  """Computes relu of x element-wise"""
  return tfe.relu(x)


def sigmoid(x):
  """Computes sigmoid of x element-wise"""
  return tfe.sigmoid(x)


def sigmoid_deriv(y, d_y):
  """Computes derive sigmoid of y"""
  return d_y * y * (tfe.negative(y) + 1)


def tanh(x):
  """Computes tanh of x element-wise"""
  return tfe.tanh(x)


def linear(x):
  return x


def get(identifier):
  """get the activation function"""
  if identifier is None:
    return linear
  if callable(identifier):
    return identifier
  if isinstance(identifier, str):
    activations = {
        "relu": relu,
        "sigmoid": sigmoid,
        "tanh": tanh,
        "linear": linear,
    }
    return activations[identifier]


def get_deriv(identifier):
  """get the activation derivative function"""
  if identifier is None:
    return linear
  if callable(identifier):
    raise NotImplementedError('During training, please use a string '
                              '(e.g "relu") to specify the activation '
                              'function instead of calling directly '
                              'the activation function.')
  if isinstance(identifier, str):
    activations = {"sigmoid": sigmoid_deriv}
    if identifier not in activations.keys():
      raise NotImplementedError('Activation function {} not yet implemented '
                                'during training'.format(identifier))
    return activations[identifier]

  raise ValueError('Could not interpret '
                   'activation function identifier:', identifier)
