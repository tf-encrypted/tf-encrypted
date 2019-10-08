"""An example of the secure aggregation protocol for federated learning."""
#pylint: disable=redefined-outer-name
#pylint:disable=unexpected-keyword-arg
from datetime import datetime
import functools
import logging
import sys

import tensorflow as tf
import tf_encrypted as tfe

from convert import decode
from util import UndefinedModelFnError

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
      'data-owner-2'
  ])

tfe.set_config(config)
tfe.set_protocol(tfe.protocol.Pond())

EPOCHS = 1
NUM_DATA_OWNERS = 3

BATCH_SIZE = 256
DATA_ITEMS = 60000
BATCHES = DATA_ITEMS // NUM_DATA_OWNERS // BATCH_SIZE

TRACING = False

def split_dataset(num_data_owners):
  """
  Helper function to help split the dataset evenly between
  data owners. USE FOR SIMULATION ONLY.
  """

  print("WARNING: Splitting dataset for {} data owners. "
        "This is for simulation use only".format(num_data_owners))

  all_dataset = tf.data.TFRecordDataset(["./data/train.tfrecord"])

  split = DATA_ITEMS // num_data_owners
  index = 0
  for i in range(num_data_owners):
    dataset = all_dataset.skip(index)
    dataset = all_dataset.take(split)

    filename = './data/train{}.tfrecord'.format(i)
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(dataset)

def build_data_pipeline(filename, batch_size=BATCH_SIZE):
  """Build data pipeline for validation by model owner."""
  def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

  dataset = tf.data.TFRecordDataset([filename])
  dataset = dataset.map(decode)
  dataset = dataset.map(normalize)
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.repeat()

  return dataset


### Owner Metaclass ###

class Owner(type):
  def __call__(self, *args, **kwargs):
    owner_obj = type.__call__(*args, **kwargs)

    # Decorate user-defined TF function that needs to be pinned to a particular
    # device and compiled
    if hasattr(owner_obj, "aggregator_fn"):
      self.handle_tf_local_fn(owner_obj, "aggregator_fn")

    if hasattr(owner_obj, "evaluator_fn"):
      self.handle_tf_local_fn(owner_obj, "evaluator_fn")

    # Decorate user-defined TFE local_computations
    if has_attr(obj, "model_fn"):
      self.handle_tfe_local_fn(owner_obj, "model_fn")

  @classmethod
  def handle_tf_local_fn(mcs, owner_obj, func_name):
    func = getattr(owner_obj, func_name)
    pinned_evaluator = self.pin_to_owner(owner_obj)(func)
    setattr(owner_obj, func_name, tf.function(pinned_evaluator))

  @classmethod
  def handle_tfe_local_fn(mcs, owner_obj, func_name):
    func = getattr(owner_obj, func_name)
    setattr(owner_obj, func_name, tfe.local_computation(func))

  @classmethod
  def pin_to_owner(mcs, owner_obj):
    def wrapper(func):

      @functools.wraps(func)
      def pinned_fn(*args, **kwargs):
        with owner_obj.device:
          return func(*args, **kwargs)

      return pinned_fn
    return wrapper


### Owner classes ###


class ModelOwner(BaseModelOwner):
  """Contains code meant to be executed by some `ModelOwner` Player.

  Args:
    player_name: `str`, name of the `tfe.player.Player`
                 representing the model owner.
  """

  def model_fn(cls, data_owner):
    return default_model_fn(data_owner)

  def aggregator_fn(cls, model_gradients):
    return secure_aggregation(model_gradients)

  def evaluator_fn(self):
    return evaluate_classifier(self)


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
  pass

if __name__ == "__main__":
  split_dataset(NUM_DATA_OWNERS)

  logging.basicConfig(level=logging.DEBUG)

  model = tf.keras.Sequential((
      tf.keras.layers.Dense(512, input_shape=[None, 28 * 28],
                            activation='relu'),
      tf.keras.layers.Dense(10),
  ))

  model.build()

  loss = tf.keras.losses.sparse_categorical_crossentropy

  model_owner = ModelOwner("model-owner", model,
                           tf.keras.optimizers.Adam(), loss)

  data_owners = [DataOwner("data-owner-{}".format(i),
                           "./data/train{}.tfrecord".format(i),
                           model,
                           loss) for i in range(NUM_DATA_OWNERS)]

  # if TRACING:
  #   stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
  #   logdir = 'logs/func/%s' % stamp
  #   writer = tf.summary.create_file_writer(logdir)
  #
  #   tf.summary.trace_on(graph=True, profiler=True)
  #
  #   for data_owner in data_owners:
  #     data_owner.model.set_weights(model_owner.model.get_weights())
  #
  #   # only run once for TRACING
  #   train_step_master(model_owner, data_owners)
  #
  #   with writer.as_default():
  #     tf.summary.trace_export(name="train_step_master", step=0,
  #                             profiler_outdir=logdir)
  # else:
  #   for i in range(EPOCHS):
  #     for i in range(BATCHES):
  #       for data_owner in data_owners:
  #         data_owner.model.set_weights(model_owner.model.get_weights())
  #
  #       if i % 10 == 0:
  #         print("Batch {}".format(i))
  #
  #       train_step_master(model_owner, data_owners)
  model_owner.fit(data_owners, rounds=BATCHES, evaluate_every=10)
