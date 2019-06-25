# pylint:  disable=redefined-outer-name
"""An example of performing secure inference with MNIST.

Also performs plaintext training.
"""

import sys

import tensorflow as tf
from tensorflow import contrib

import tensorflow.keras as keras

import tf_encrypted as tfe

from convert import decode


if len(sys.argv) > 1:
  # config file was specified
  config_file = sys.argv[1]
  config = tfe.RemoteConfig.load(config_file)
  tfe.set_config(config)
  tfe.set_protocol(tfe.protocol.Pond())

session_target = sys.argv[2] if len(sys.argv) > 2 else None


class ModelOwner():
  """
  Contains code meant to be executed by the model owner.

  Args:
    player_name: `str`, name of the `tfe.player.Player`
                 representing the model owner.
  """
  
  BATCH_SIZE = 128
  NUM_CLASSES = 10
  EPOCHS = 2

  IMG_ROWS, IMG_COLS = 28, 28

  def __init__(self, player_name, local_data_file):
    self.player_name = player_name
    self.local_data_file = local_data_file

  def _build_data_pipeline(self):
    """Build a reproducible tf.data iterator."""

    def normalize(image, label):
      image = tf.cast(image, tf.float32) / 255.0
      return image, label

    dataset = tf.data.TFRecordDataset([self.local_data_file])
    dataset = dataset.map(decode)
    dataset = dataset.map(normalize)
    dataset = dataset.repeat()
    dataset = dataset.batch(self.BATCH_SIZE)

    iterator = dataset.make_one_shot_iterator()
    return iterator

  def _build_training_graph(self, training_data):
    """Build a graph for plaintext model training."""

    model = keras.Sequential()
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, input_shape=[28, 28, 1]))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(self.NUM_CLASSES, activation='softmax'))

    def loss(model, x, y):
      y_ = model(x)
      return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

    def grad(model, inputs, targets):
      with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
      return loss_value, tape.gradient(loss_value, model.trainable_variables)

    # optimizer and data pipeline
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    global_step = tf.Variable(0)
    for epoch in range(self.EPOCHS):
      # Training loop - using batches of 32
      for x, y in training_data.get_next():
        # Optimize the model
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                  global_step)

    return model.trainable_weights

  def provide_input(self):
    with tf.name_scope('loading'):
      training_data = self._build_data_pipeline()

    with tf.name_scope('training'):
      parameters = self._build_training_graph(training_data)

    return parameters


class PredictionClient():
  """
  Contains code meant to be executed by a prediction client.

  Args:
    player_name: `str`, name of the `tfe.player.Player`
                 representing the data owner
    build_update_step: `Callable`, the function used to construct
                       a local federated learning update.
  """

  BATCH_SIZE = 20

  def __init__(self, player_name, local_data_file):
    self.player_name = player_name
    self.local_data_file = local_data_file

  def _build_data_pipeline(self):
    """Build a reproducible tf.data iterator."""

    def normalize(image, label):
      image = tf.cast(image, tf.float32) / 255.0
      return image, label

    dataset = tf.data.TFRecordDataset([self.local_data_file])
    dataset = dataset.map(decode)
    dataset = dataset.map(normalize)
    dataset = dataset.repeat()
    dataset = dataset.batch(self.BATCH_SIZE)

    iterator = dataset.make_one_shot_iterator()
    return iterator

  def provide_input(self) -> tf.Tensor:
    """Prepare input data for prediction."""
    with tf.name_scope('loading'):
      prediction_input, expected_result = self._build_data_pipeline().get_next()
      print_op = tf.print("Expect", expected_result, summarize=self.BATCH_SIZE)
      with tf.control_dependencies([print_op]):
        prediction_input = tf.identity(prediction_input)

    with tf.name_scope('pre-processing'):
      prediction_input = tf.reshape(
          prediction_input, shape=(self.BATCH_SIZE, 28 * 28))
    return prediction_input

  def receive_output(self, logits: tf.Tensor) -> tf.Operation:
    with tf.name_scope('post-processing'):
      prediction = tf.argmax(logits, axis=1)
      op = tf.print("Result", prediction, summarize=self.BATCH_SIZE)
      return op


if __name__ == "__main__":

  model_owner = ModelOwner(
      player_name="model-owner",
      local_data_file="./data/train.tfrecord")

  prediction_client = PredictionClient(
      player_name="prediction-client",
      local_data_file="./data/test.tfrecord")

  # get model parameters as private tensors from model owner
  params = tfe.define_private_variable(model_owner.player_name,
                                    model_owner.provide_input)  # pylint: disable=E0632

  # we'll use the same parameters for each prediction so we cache them to
  # avoid re-training each time
  cache_updater, params = tfe.cache(params)

  w0, b0, w1, b1 = params
  model = tfe.keras.Sequential([
    tfe.keras.layers.Dense(512, batch_input_shape=(PredictionClient.BATCH_SIZE, 784)),
    tfe.keras.layers.Activation('relu'),
    tfe.keras.layers.Dense(10, activation=softmax)
  ])

  model.set_weights([w0,w1])

  # get prediction input from client
  x = tfe.define_private_input(prediction_client.player_name,
                               prediction_client.provide_input, masked=True)  # pylint: disable=E0632
  
  logits = model(x)

  # send prediction output back to client
  prediction_op = tfe.define_output(prediction_client.player_name,
                                    logits,
                                    prediction_client.receive_output)

  with tfe.Session(target=session_target) as sess:

    sess.run(tf.global_variables_initializer(), tag='init')

    print("Training")
    sess.run(cache_updater, tag='training')

    for _ in range(5):
      print("Predicting")
      sess.run(prediction_op, tag='prediction')
