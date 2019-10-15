"""An example of the secure aggregation protocol for federated learning."""
#pylint: disable=redefined-outer-name
#pylint:disable=unexpected-keyword-arg
#pylint:disable=wrong-import-position
import argparse
import logging

from absl import app
from absl import flags
import tensorflow as tf
import tf_encrypted as tfe

from players import BaseModelOwner, BaseDataOwner
from func_lib import (
    default_model_fn,
    secure_mean,
    evaluate_classifier,
    secure_reptile,
    reptile_model_fn,
)

# Data flags
flags.DEFINE_string("remote_config", None, "Specify remote configuration")
flags.DEFINE_string("data_root", "./data",
                    "Specify the root directory of the data")
flags.DEFINE_integer("data_items", 60000, "Length of dataset")
flags.DEFINE_integer("num_data_owners", 3,
                     ("Specify how many data owners should "
                      "take part in the learning. If remote "
                      "configuration is specified the number of data owners "
                      "must match the number specified here"))

# Learning flags
flags.DEFINE_float("learning_rate", .01, "Global learning rate")
flags.DEFINE_float("local_learning_rate", None,
                   ("Local learning rate (only used for reptile learning "
                    "rule). If None, defaults to `learning_rate`."))
flags.DEFINE_boolean("reptile", False,
                     ("If True, the ModelOwner will use the Reptile "
                      "meta-learning algorithm for performing updates to the "
                      "master model."))
flags.DEFINE_integer("epochs", 3, ("Number of epochs - used with number of "
                                   "data owners & batch size to determine "
                                   "total number of iterations"))
flags.DEFINE_integer("batch_size", 100, "Batch size")

# Misc flags
flags.DEFINE_boolean("split", True,
                     ("Whether the script runner should help "
                      "simulate the training by splitting the data and "
                      "distributing it amongst the data owners. "
                      "Only applicable for local computations."))

FLAGS = flags.FLAGS

def split_dataset(num_data_owners):
  """
  Helper function to help split the dataset evenly between
  data owners. USE FOR SIMULATION ONLY.
  """

  print("WARNING: Splitting dataset for {} data owners. "
        "This is for simulation use only".format(num_data_owners))

  all_dataset = tf.data.TFRecordDataset([FLAGS.data_root + "/train.tfrecord"])

  split = FLAGS.data_items // num_data_owners
  index = 0
  for i in range(num_data_owners):
    dataset = all_dataset.skip(index)
    dataset = all_dataset.take(split)

    filename = '{}/train{}.tfrecord'.format(FLAGS.data_root, i)
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(dataset)

### Owner classes ###
class ModelOwner(BaseModelOwner):
  """Contains code meant to be executed by some `ModelOwner` Player.

  Args:
    player_name: `str`, name of the `tfe.player.Player`
                 representing the model owner.
  """
  @classmethod
  def model_fn(cls, data_owner):
    if FLAGS.reptile:
      return reptile_model_fn(data_owner)

    return default_model_fn(data_owner)

  @classmethod
  def aggregator_fn(cls, model_gradients, model):
    if FLAGS.reptile:
      return secure_reptile(model_gradients, model)

    return secure_mean(model_gradients)

  @classmethod
  def evaluator_fn(cls, model_owner):
    return evaluate_classifier(model_owner)


class DataOwner(BaseDataOwner):
  """Contains methods meant to be executed by a data owner.

  Args:
    player_name: `str`, name of the `tfe.player.Player`
                 representing the data owner
    build_update_step: `Callable`, the function used to construct
                       a local federated learning update.
  """
  # TODO: stick `build_data_pipeline` somewhere in here
  # TODO: can also move model_fn in here -- we leave it up to the user atm


def main(_):
  if FLAGS.remote_config is not None:
    config = tfe.RemoteConfig.load(FLAGS.remote_config)
    config.connect_to_cluster()
  else:
    players = ['server0', 'server1', 'crypto-producer', 'model-owner']
    data_owners = ['data-owner-{}'.format(i) for i in range(FLAGS.num_data_owners)]
    config = tfe.EagerLocalConfig(players + data_owners)

  tfe.set_config(config)
  tfe.set_protocol(tfe.protocol.Pond())

  NUM_DATA_OWNERS = FLAGS.num_data_owners

  BATCH_SIZE = FLAGS.batch_size
  DATA_ITEMS = 60000
  BATCHES = DATA_ITEMS // NUM_DATA_OWNERS // BATCH_SIZE
  REPTILE = FLAGS.reptile

  if FLAGS.split:
    split_dataset(NUM_DATA_OWNERS)

  logging.basicConfig(level=logging.DEBUG)

  model = tf.keras.Sequential((
      tf.keras.layers.Dense(512, input_shape=[None, 28 * 28],
                            activation='relu'),
      tf.keras.layers.Dense(10),
  ))

  model.build()

  loss = tf.keras.losses.sparse_categorical_crossentropy

  local_lr = FLAGS.local_learning_rate or FLAGS.learning_rate

  model_owner = ModelOwner("model-owner",
                           "{}/train.tfrecord".format(FLAGS.data_root),
                           model, loss,
                           optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate))

  data_owners = [DataOwner("data-owner-{}".format(i),
                           "{}/train{}.tfrecord".format(FLAGS.data_root, i),
                           model, loss,
                           optimizer=tf.keras.optimizers.Adam(local_lr))
                 for i in range(NUM_DATA_OWNERS)]

  model_owner.fit(data_owners, rounds=BATCHES, evaluate_every=10)

if __name__ == "__main__":
  app.run(main)
