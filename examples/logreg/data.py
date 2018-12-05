from typing import Tuple

import numpy as np
import tensorflow as tf

np.random.seed(1)


def norm(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.cast(x, tf.float32), tf.expand_dims(y, 0)


def gen_training_input(total_size: int, nb_feats: int, bs: int) -> Tuple[tf.Tensor, tf.Tensor]:
    x_np = np.random.uniform(-.5, .5, size=[total_size, nb_feats])
    y_np = np.array(x_np.mean(axis=1) > 0, np.float32)
    train_set = tf.data.Dataset.from_tensor_slices((x_np, y_np)) \
                               .map(norm) \
                               .shuffle(buffer_size=100) \
                               .repeat() \
                               .batch(bs)
    train_set_iterator = train_set.make_one_shot_iterator()
    x, y = train_set_iterator.get_next()
    x = tf.reshape(x, [bs, nb_feats])
    y = tf.reshape(y, [bs, 1])

    # return tf.print(x, data=[x], message="x: ", summarize=6), y
    return x, y


def gen_test_input(total_size: int, nb_feats: int, bs: int) -> Tuple[tf.Tensor, tf.Tensor, np.ndarray]:
    x_test_np = np.random.uniform(-.5, .5, size=[total_size, nb_feats])
    y_test_np = np.array(x_test_np.mean(axis=1) > 0, np.float32)
    test_set = tf.data.Dataset.from_tensor_slices((x_test_np, y_test_np)) \
                              .map(norm) \
                              .batch(bs)
    test_set_iterator = test_set.make_one_shot_iterator()
    x_test, y_test = test_set_iterator.get_next()
    x_test = tf.reshape(x_test, [bs, nb_feats])
    y_test = tf.reshape(y_test, [bs, 1])

    return x_test, y_test
