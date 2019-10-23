""" Example utils """

import functools

class UndefinedModelFnError(Exception):
  """
  UndefinedModelFnError

  Occurs if a data owner or model owner hasn't defined
  a model function
  """

def pin_to_owner(func):

  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    with args[0].device:
      return func(*args, **kwargs)

    return pinned_fn

  return wrapper
