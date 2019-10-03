"""An example of the secure aggregation protocol for federated learning."""

import sys
import logging

import tensorflow as tf
import tf_encrypted as tfe

from convert import decode

if len(sys.argv) > 1:
  # config file was specified
  config_file = sys.argv[1]
  config = tfe.RemoteConfig.load(config_file)
  config.connect_to_cluster()
else:
  config = tfe.EagerLocalConfig([
      'server0',
      'server1',
      'crypto-producer',
      'model-owner',
      'data-owner-0',
      'data-owner-1',
      'data-owner-2',
  ])

tfe.set_config(config)
tfe.set_protocol(tfe.protocol.Pond())

EPOCHS = 1
BATCH_SIZE = 256
BATCHES = 60000 // BATCH_SIZE

@tfe.local_computation
def build_data_pipeline(validation=False, batch_size=BATCH_SIZE):
  """Build data pipeline for validation by model owner."""
  def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

  dataset = tf.data.TFRecordDataset(["./data/train.tfrecord"])
  dataset = dataset.map(decode)
  dataset = dataset.map(normalize)
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.repeat()

  if validation:
    dataset = dataset.take(1)  # keep validating on the same items

  return dataset

class ModelOwner:
  """Contains code meant to be executed by some `ModelOwner` Player.

  Args:
    player_name: `str`, name of the `tfe.player.Player`
                 representing the model owner.
  """

  def __init__(self, player_name, model, optimizer, loss):
    self.player_name = player_name

    self.model = model
    self.optimizer = optimizer
    self.loss = loss

    self.dataset = iter(build_data_pipeline(validation=True, batch_size=50,
                                            player_name=self.player_name))

class DataOwner:
  """Contains methods meant to be executed by a data owner.

  Args:
    player_name: `str`, name of the `tfe.player.Player`
                 representing the data owner
    build_update_step: `Callable`, the function used to construct
                       a local federated learning update.
  """
  def __init__(self, player_name, local_data_file, loss):
    self.player_name = player_name
    self.local_data_file = local_data_file
    self.loss = loss

    self.dataset = iter(build_data_pipeline(player_name=self.player_name))

@tfe.local_computation
def update_model(model_owner, *grads):
  """Perform a single update step.

  This will be performed on the ModelOwner device
  after securely aggregating gradients.

  Args:
    *grads: `tf.Variables` representing the federally computed gradients.
  """
  params = model_owner.model.trainable_variables
  grads = [tf.cast(grad, tf.float32) for grad in grads]
  with tf.name_scope('update'):
    model_owner.optimizer.apply_gradients(zip(grads, model_owner.model.trainable_variables))

  return grads

def securely_aggregate(model_grads):
  with tf.name_scope('secure_aggregation'):
    aggregated_model_grads = [
        tfe.add_n(grads) / len(grads)
        for grads in model_grads
    ]

  return aggregated_model_grads

@tfe.local_computation
def validation_step(model_owner):
  x, y = next(model_owner.dataset)

  with tf.name_scope('validate'):
    predictions = model_owner.model(x)
    loss = tf.reduce_mean(model_owner.loss(y, predictions, from_logits=True))

    y_hat = tf.argmax(input=predictions, axis=1)

  tf.print("loss", loss)
  tf.print("expect", y, summarize=50)
  tf.print("result", y_hat, summarize=50)
  return loss, y, y_hat

@tfe.local_computation
def train_step(data_owner):
  x, y = next(data_owner.dataset)

  with tf.name_scope('gradient_computation'):
    with tf.GradientTape() as tape:
      predictions = model(x)
      loss = tf.reduce_mean(data_owner.loss(y, predictions, from_logits=True))

    grads = tape.gradient(loss, model.trainable_variables)

  return grads


@tf.function
def train_step_master(model_owner, data_owners):
  grads = []

  for data_owner in data_owners:
    grads.append(train_step(data_owner, player_name=data_owner.player_name))

  agg_grads = securely_aggregate(zip(*grads))

  update_model(model_owner, *agg_grads, player_name=model_owner.player_name)
  validation_step(model_owner, player_name=model_owner.player_name)

if __name__ == "__main__":

  logging.basicConfig(level=logging.DEBUG)

  model = tf.keras.Sequential((
      tf.keras.layers.Dense(512, input_shape=[None, 28 * 28], activation='sigmoid'),
      tf.keras.layers.Dense(10),
  ))

  model.build()

  loss = tf.keras.losses.sparse_categorical_crossentropy

  model_owner = ModelOwner("model-owner", model, tf.keras.optimizers.Adam(), loss)
  data_owners = [
      DataOwner("data-owner-0", "./data/train.tfrecord", loss),
      DataOwner("data-owner-1", "./data/train.tfrecord", loss),
      DataOwner("data-owner-2", "./data/train.tfrecord", loss),
  ]

  for i in range(EPOCHS):
    for i in range(BATCHES):
      if i % 100 == 0:
        print("Batch {}".format(i))

      train_step_master(model_owner, data_owners)
