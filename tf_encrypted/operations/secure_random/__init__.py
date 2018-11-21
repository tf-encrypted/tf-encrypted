import tensorflow as tf
from .secure_random import seeded_secure_random

# FEATURE FLAG: seeded_secure_random or tf.random_uniform
random_uniform = tf.random_uniform

__all__ = [
    "seeded_secure_random"
]
