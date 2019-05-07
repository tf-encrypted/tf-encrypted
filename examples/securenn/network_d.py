# pylint:  disable=redefined-outer-name
"""An example of performing secure inference with MNIST.

Reproduces Network D from SecureNN, Wagh et al.
"""
from __future__ import absolute_import
import math
import sys
from typing import List, Tuple

import tensorflow as tf
import tf_encrypted as tfe

from examples.mnist.convert import get_data_from_tfrecord

# tfe.set_tfe_events_flag(True)

if len(sys.argv) >= 2:
  # config file was specified
  config_file = sys.argv[1]
  config = tfe.config.load(config_file)
else:
  # default to using local config
  config = tfe.LocalConfig([
      'server0',
      'server1',
      'crypto-producer',
      'model-trainer',
      'prediction-client'
  ])
tfe.set_config(config)
tfe.set_protocol(tfe.protocol.SecureNN(
    *tfe.get_config().get_players(['server0', 'server1', 'crypto-producer'])))


def weight_variable(shape, gain):
  """weight_variable generates a weight variable of a given shape."""
  if len(shape) == 2:
    fan_in, fan_out = shape
  elif len(shape) == 4:
    h, w, c_in, c_out = shape
    fan_in = h * w * c_in
    fan_out = h * w * c_out
  r = gain * math.sqrt(6 / (fan_in + fan_out))
  initial = tf.random_uniform(shape, minval=-r, maxval=r)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0., shape=shape)
  return tf.Variable(initial)


def conv2d(x, w, s):
  return tf.nn.conv2d(x, w, strides=[1, s, s, 1], padding='VALID')


def pooling(x):
  return tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')


class ModelTrainer():
  """Contains code meant to be executed by a model training Player."""

  BATCH_SIZE = 256
  ITERATIONS = 60000 // BATCH_SIZE
  EPOCHS = 3
  LEARNING_RATE = 3e-3
  IN_DIM = 28
  KERNEL = 5
  STRIDE = 2
  IN_CHANNELS = 1
  HIDDEN_CHANNELS = 5
  HIDDEN_FC1 = 180
  HIDDEN_FC2 = 100
  OUT_N = 10

  def cond(self,
           i: tf.Tensor,
           max_iter: tf.Tensor,
           nb_epochs: tf.Tensor,
           avg_loss: tf.Tensor):
    """Check if training termination condition has been met."""
    is_end_epoch = tf.equal(i % max_iter, 0)
    to_continue = tf.cast(i < max_iter * nb_epochs, tf.bool)

    def true_fn() -> tf.Tensor:
      to_continue = tf.print("avg_loss: ", avg_loss)
      return to_continue

    def false_fn() -> tf.Tensor:
      return to_continue

    return tf.cond(is_end_epoch, true_fn, false_fn)

  def build_training_graph(self, training_data) -> List[tf.Tensor]:
    """Build a graph for plaintext model training.

    Returns a list of the trained model's parameters.
    """
    # model parameters and initial values
    wconv1 = weight_variable([self.KERNEL,
                              self.KERNEL,
                              self.IN_CHANNELS,
                              self.HIDDEN_CHANNELS], 1.)
    bconv1 = bias_variable([1, 1, self.HIDDEN_CHANNELS])
    wfc1 = weight_variable([self.HIDDEN_FC1, self.HIDDEN_FC2], 1.)
    bfc1 = bias_variable([self.HIDDEN_FC2])
    wfc2 = weight_variable([self.HIDDEN_FC2, self.OUT_N], 1.)
    bfc2 = bias_variable([self.OUT_N])
    params = [wconv1, bconv1, wfc1, bfc1, wfc2, bfc2]

    # optimizer and data pipeline
    optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE)

    # training loop
    def loop_body(i: tf.Tensor,
                  max_iter: tf.Tensor,
                  nb_epochs: tf.Tensor,
                  avg_loss: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
      """Main model training loop."""
      # get next batch
      x, y = training_data.get_next()

      # model construction
      x = tf.reshape(x, [-1, self.IN_DIM, self.IN_DIM, 1])
      layer1 = pooling(tf.nn.relu(conv2d(x, Wconv1, self.STRIDE) + bconv1))
      layer1 = tf.reshape(layer1, [-1, self.HIDDEN_FC1])
      layer2 = tf.nn.relu(tf.matmul(layer1, Wfc1) + bfc1)
      logits = tf.matmul(layer2, Wfc2) + bfc2

      loss = tf.reduce_mean(
          tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=y))

      is_end_epoch = tf.equal(i % max_iter, 0)

      def true_fn() -> tf.Tensor:
        return loss

      def false_fn() -> tf.Tensor:
        prev_loss = tf.cast(i - 1, tf.float32) * avg_loss
        return (prev_loss + loss) / tf.cast(i, tf.float32)

      with tf.control_dependencies([optimizer.minimize(loss)]):
        terminal_cond = tf.cond(is_end_epoch, true_fn, false_fn)
        return i + 1, max_iter, nb_epochs, terminal_cond

    loop, _, _, _ = tf.while_loop(
        self.cond, loop_body, [0, self.ITERATIONS, self.EPOCHS, 0.])

    # return model parameters after training
    loop = tf.print("Training complete", loop)
    with tf.control_dependencies([loop]):
      return [param.read_value() for param in params]

  def provide_input(self) -> List[tf.Tensor]:
    with tf.name_scope('loading'):
      training_data = get_data_from_tfrecord(
          "./data/train.tfrecord", self.BATCH_SIZE)

    with tf.name_scope('training'):
      parameters = self.build_training_graph(training_data)

    return parameters


class PredictionClient():
  """Contains methods meant to be executed by a prediction client.

  Args:
    player_name: `str`, name of the `tfe.player.Player`
                 representing the data owner
    build_update_step: `Callable`, the function used to construct
                       a local federated learning update.
  """

  BATCH_SIZE = 20

  def provide_input(self) -> List[tf.Tensor]:
    """Prepare input data for prediction."""
    with tf.name_scope('loading'):
      prediction_input, expected_result = get_data_from_tfrecord(
          "./data/test.tfrecord", self.BATCH_SIZE).get_next()

    with tf.name_scope('pre-processing'):
      prediction_input = tf.reshape(
          prediction_input, shape=(self.BATCH_SIZE, 1, 28, 28))
      expected_result = tf.reshape(expected_result, shape=(self.BATCH_SIZE,))

    return [prediction_input, expected_result]

  def receive_output(self, likelihoods: tf.Tensor, y_true: tf.Tensor):
    with tf.name_scope('post-processing'):
      prediction = tf.argmax(likelihoods, axis=1)
      eq_values = tf.equal(prediction, tf.cast(y_true, tf.int64))
      acc = tf.reduce_mean(tf.cast(eq_values, tf.float32))
      op = tf.print('Expected:', y_true, '\nActual:',
                    prediction, '\nAccuracy:', acc)

      return op


if __name__ == '__main__':
  model_trainer = ModelTrainer()
  prediction_client = PredictionClient()

  # get model parameters as private tensors from model owner
  params = tfe.define_private_input(
      'model-trainer', model_trainer.provide_input, masked=True)  # pylint: disable=E0632

  # we'll use the same parameters for each prediction so we cache them to avoid re-training each time
  params = tfe.cache(params)

  # get prediction input from client
  x, y = tfe.define_private_input(
      'prediction-client', prediction_client.provide_input, masked=True)  # pylint: disable=E0632

  # helpers


  def conv(x, w, s):
    return tfe.conv2d(x, w, s, 'VALID')


  def pool(x):
    return tfe.avgpool2d(x, (2, 2), (2, 2), 'VALID')


  # compute prediction
  wconv1, bconv1, wfc1, bfc1, wfc2, bfc2 = params
  bconv1 = tfe.reshape(bconv1, [-1, 1, 1])
  layer1 = pool(tfe.relu(conv(x, wconv1, ModelTrainer.STRIDE) + bconv1))
  layer1 = tfe.reshape(layer1, [-1, ModelTrainer.HIDDEN_FC1])
  layer2 = tfe.matmul(layer1, wfc1) + bfc1
  logits = tfe.matmul(layer2, wfc2) + bfc2

  # send prediction output back to client
  prediction_op = tfe.define_output(
      'prediction-client', [logits, y], prediction_client.receive_output)


  with tfe.Session() as sess:
    print("Init")
    sess.run(tf.global_variables_initializer(), tag='init')

    print("Training")
    sess.run(tfe.global_caches_updater(), tag='training')

    for _ in range(5):
      print("Predicting")
      sess.run(prediction_op, tag='prediction')
