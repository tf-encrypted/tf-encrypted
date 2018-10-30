import tensorflow as tf
from .secure_random import secure_random

# FEATURE FLAG: secure_random or tf.random_uniform
random_func = tf.random_uniform

__all__ = [
    "secure_random"
]
