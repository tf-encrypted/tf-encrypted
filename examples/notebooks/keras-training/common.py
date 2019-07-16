"""Provide classes to perform private training and private prediction with
logistic regression"""

import tensorflow as tf
import tf_encrypted as tfe


class DataOwner:
  """Contains code meant to be executed by a data owner Player."""
  def __init__(
      self,
      player_name,
      num_features,
      training_set_size,
      test_set_size,
      batch_size
  ):
    self.player_name = player_name
    self.num_features = num_features
    self.training_set_size = training_set_size
    self.test_set_size = test_set_size
    self.batch_size = batch_size
    self.train_initializer = None
    self.test_initializer = None

  @property
  def initializer(self):
    return tf.group(self.train_initializer, self.test_initializer)

  @tfe.local_computation
  def provide_training_data(self):
    """Preprocess training dataset

    Return single batch of training dataset
    """
    def norm(x, y):
      return tf.cast(x, tf.float32), tf.expand_dims(y, 0)

    x_raw = tf.random.uniform(
        minval=-.5,
        maxval=.5,
        shape=[self.training_set_size, self.num_features])

    y_raw = tf.cast(tf.reduce_mean(x_raw, axis=1) > 0, dtype=tf.float32)

    train_set = tf.data.Dataset.from_tensor_slices((x_raw, y_raw)) \
        .map(norm) \
        .repeat() \
        .shuffle(buffer_size=self.batch_size) \
        .batch(self.batch_size)

    train_set_iterator = train_set.make_initializable_iterator()
    self.train_initializer = train_set_iterator.initializer

    x, y = train_set_iterator.get_next()
    x = tf.reshape(x, [self.batch_size, self.num_features])
    y = tf.reshape(y, [self.batch_size, 1])

    return x, y

  @tfe.local_computation
  def provide_testing_data(self):
    """Preprocess testing dataset

    Return single batch of testing dataset
    """
    def norm(x, y):
      return tf.cast(x, tf.float32), tf.expand_dims(y, 0)

    x_raw = tf.random.uniform(
        minval=-.5,
        maxval=.5,
        shape=[self.test_set_size, self.num_features])

    y_raw = tf.cast(tf.reduce_mean(x_raw, axis=1) > 0, dtype=tf.float32)

    test_set = tf.data.Dataset.from_tensor_slices((x_raw, y_raw)) \
        .map(norm) \
        .batch(self.test_set_size)

    test_set_iterator = test_set.make_initializable_iterator()
    self.test_initializer = test_set_iterator.initializer

    x, y = test_set_iterator.get_next()
    x = tf.reshape(x, [self.test_set_size, self.num_features])
    y = tf.reshape(y, [self.test_set_size, 1])

    return x, y
