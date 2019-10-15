"""An example of the secure aggregation protocol for federated learning."""
#pylint: disable=redefined-outer-name
#pylint:disable=unexpected-keyword-arg
#pylint:disable=wrong-import-position
import logging
import argparse

import tensorflow as tf
import tf_encrypted as tfe

parser = argparse.ArgumentParser(description="Federated learning example")
parser.add_argument("--remote-config", type=str,
                    default=None, help="Specify remote configuration")
parser.add_argument("--data-root", type=str,
                    default="./data",
                    help="Specify the root directory of the data")
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--num-data-owners", type=int,
                    default=3,
                    help="Specify how many data owners should "
                         "take part in the learning. If remote "
                         "configuration is specified the number of data owners "
                         "must match the number specified here")
parser.add_argument("--no-split", action="store_true", default=False,
                    help="Whether the script runner should help "
                         "simulate the training by splitting the data and "
                         "distributing it amongst the data owners. "
                         "Only applicable for local computations.")
parser.add_argument("--reptile", action="store_true", default=False,
                    help="If set the ModelOwner will use the reptile "
                         "method of calculating gradients "
                         "and aggregating them.")

args = parser.parse_args()

if args.remote_config is not None:
  config = tfe.RemoteConfig.load(args.remote_config)
  config.connect_to_cluster()
else:
  players = ['server0', 'server1', 'crypto-producer', 'model-owner']
  data_owners = ['data-owner-{}'.format(i) for i in range(args.num_data_owners)]
  config = tfe.EagerLocalConfig(players + data_owners)

tfe.set_config(config)
tfe.set_protocol(tfe.protocol.Pond())

from players import BaseModelOwner, BaseDataOwner
from func_lib import default_model_fn, secure_mean, evaluate_classifier, secure_reptile, reptile_model_fn

NUM_DATA_OWNERS = args.num_data_owners

BATCH_SIZE = args.batch_size
DATA_ITEMS = 60000
BATCHES = DATA_ITEMS // NUM_DATA_OWNERS // BATCH_SIZE
REPTILE = args.reptile

def split_dataset(num_data_owners):
  """
  Helper function to help split the dataset evenly between
  data owners. USE FOR SIMULATION ONLY.
  """

  print("WARNING: Splitting dataset for {} data owners. "
        "This is for simulation use only".format(num_data_owners))

  all_dataset = tf.data.TFRecordDataset([args.data_root + "/train.tfrecord"])

  split = DATA_ITEMS // num_data_owners
  index = 0
  for i in range(num_data_owners):
    dataset = all_dataset.skip(index)
    dataset = all_dataset.take(split)

    filename = '{}/train{}.tfrecord'.format(args.data_root, i)
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
    if REPTILE:
      return reptile_model_fn(data_owner)

    return default_model_fn(data_owner)

  @classmethod
  def aggregator_fn(cls, model_gradients, model):
    if REPTILE:
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

if __name__ == "__main__":
  if not args.no_split:
    split_dataset(NUM_DATA_OWNERS)

  logging.basicConfig(level=logging.DEBUG)

  model = tf.keras.Sequential((
      tf.keras.layers.Dense(512, input_shape=[None, 28 * 28],
                            activation='relu'),
      tf.keras.layers.Dense(10),
  ))

  model.build()

  loss = tf.keras.losses.sparse_categorical_crossentropy

  model_owner = ModelOwner("model-owner",
                           "{}/train.tfrecord".format(args.data_root),
                           model, loss,
                           optimizer=tf.keras.optimizers.Adam())

  data_owners = [DataOwner("data-owner-{}".format(i),
                           "{}/train{}.tfrecord".format(args.data_root, i),
                           model, loss,
                           optimizer=tf.keras.optimizers.Adam())
                 for i in range(NUM_DATA_OWNERS)]

  model_owner.fit(data_owners, rounds=BATCHES, evaluate_every=10)
