"""Provide activation functions"""
from tf_encrypted import get_protocol


def relu(x):
  """Computes relu of x element-wise"""
  return get_protocol().relu(x)

def sigmoid(x):
  """Computes sigmoid of x element-wise"""
  return get_protocol().sigmoid(x)

def tanh(x):
  """Computes tanh of x element-wise"""
  return get_protocol().tanh(x)

def linear(x):
  return x

def get(identifier):
  """get the activation function"""
  if identifier is None:
    return linear
  if callable(identifier):
    return identifier
  if isinstance(identifier, str):
    activations = {"relu": relu,
                   "sigmoid": sigmoid,
                   "tanh": tanh,
                   "linear": linear}
    return activations[identifier]

  raise ValueError('Could not interpret '
                   'activation function identifier:', identifier)
