"""Provide classes to perform private training and private prediction with
logistic regression"""
import tensorflow as tf
import tf_encrypted as tfe


class LogisticRegression:
  """Contains methods to build and train logistic regression."""
  def __init__(self, num_features):
    self.w = tfe.Variable(
        tf.random_uniform([num_features, 1], -0.01, 0.01))
    self.w_masked = tfe.mask(self.w)
    self.b = tfe.Variable(tf.zeros([1]))
    self.b_masked = tfe.mask(self.b)

  @property
  def weights(self):
    return self.w, self.b

  def forward(self, x):
    with tf.name_scope("forward"):
      out = tfe.matmul(x, self.w_masked) + self.b_masked
      y = tfe.sigmoid(out)
      return y

  def backward(self, x, dy, learning_rate=0.01):
    batch_size = x.shape.as_list()[0]
    with tf.name_scope("backward"):
      dw = tfe.matmul(tfe.transpose(x), dy) / batch_size
      db = tfe.reduce_sum(dy, axis=0) / batch_size
      assign_ops = [
          tfe.assign(self.w, self.w - dw * learning_rate),
          tfe.assign(self.b, self.b - db * learning_rate),
      ]
      return assign_ops

  def loss_grad(self, y, y_hat):
    with tf.name_scope("loss-grad"):
      dy = y_hat - y
      return dy

  def fit_batch(self, x, y):
    with tf.name_scope("fit-batch"):
      y_hat = self.forward(x)
      dy = self.loss_grad(y, y_hat)
      fit_batch_op = self.backward(x, dy)
      return fit_batch_op

  def fit(self, sess, x, y, num_batches):
    fit_batch_op = self.fit_batch(x, y)
    for batch in range(num_batches):
      print("Batch {0: >4d}".format(batch))
      sess.run(fit_batch_op, tag='fit-batch')

  def evaluate(self, sess, x, y, data_owner):
    """Return the accuracy"""
    def print_accuracy(y_hat, y) -> tf.Operation:
      with tf.name_scope("print-accuracy"):
        correct_prediction = tf.equal(tf.round(y_hat), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print_op = tf.print("Accuracy on {}:".format(data_owner.player_name),
                            accuracy)
        return print_op

    with tf.name_scope("evaluate"):
      y_hat = self.forward(x)
      print_accuracy_op = tfe.define_output(data_owner.player_name,
                                            [y_hat, y],
                                            print_accuracy)

    sess.run(print_accuracy_op, tag='evaluate')


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


class ModelOwner:
  """Contains code meant to be executed by a model owner Player."""
  def __init__(self, player_name):
    self.player_name = player_name

  @tfe.local_computation
  def receive_weights(self, *weights):
    return tf.print("Weights on {}:".format(self.player_name), weights)


class PredictionClient:
  """Contains methods meant to be executed by a prediction client."""
  def __init__(self, player_name, num_features):
    self.player_name = player_name
    self.num_features = num_features

  @tfe.local_computation
  def provide_input(self):
    return tf.random.uniform(
        minval=-.5,
        maxval=.5,
        dtype=tf.float32,
        shape=[1, self.num_features])

  @tfe.local_computation
  def receive_output(self, result):
    return tf.print("Result on {}:".format(self.player_name), result)
