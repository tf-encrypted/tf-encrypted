# pylint: disable=arguments-differ
"""Utilities for layers"""


def normalize_data_format(value):
  if value is None:
    value = 'channels_last'
  data_format = value.lower()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('The `data_format` argument must be one of '
                     '"channels_first", "channels_last". Received: ' +
                     str(value))
  return data_format
