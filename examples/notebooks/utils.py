"""
Various helpers that make using TensorFlow and TF Encrypted in notebooks
easier.
"""

import tensorflow as tf


def print_in_notebook(x):
    return tf.py_func(print, [x], Tout=[])
