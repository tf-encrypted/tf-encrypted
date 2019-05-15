"""Provide activation functions"""
from tf_encrypted.protocol.protocol import get_protocol


def prot():
  return get_protocol()

def relu(x):
  """Computes relu of x element-wise"""
  return prot().relu(x)

def sigmoid(x):
  """Computes sigmoid of x element-wise"""
  return prot().sigmoid(x)

def tanh(x):
  """Computes tanh of x element-wise"""
  return prot().tanh(x)

def linear(x):
  return x

def get(identifier):
  """get the activation function"""
  if identifier is None:
    return linear
  if isinstance(identifier, str):
    if identifier == 'relu':
      return relu
    elif identifier == 'sigmoid':
      return sigmoid
    elif identifier == 'tanh':
      return tanh
    elif identifier == 'linear':
      return linear
    else:
      raise ValueError('Could not interpret '
                       'activation function identifier:',
                       identifier)
  elif callable(identifier):
    return identifier
  else:
    raise ValueError('Could not interpret '
                     'activation function identifier:', identifier)
