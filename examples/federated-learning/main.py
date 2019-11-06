"""An example of the secure aggregation protocol for federated learning."""
# pylint: disable=redefined-outer-name
# pylint: disable=unexpected-keyword-arg
# pylint: disable=wrong-import-position
# pylint: disable=arguments-differ
# pylint: disable=abstract-method
import logging

from absl import app
from absl import flags
import tensorflow as tf
import tf_encrypted as tfe

from data_lib import federate_dataset, from_simulated_dataset
from players import BaseModelOwner, BaseDataOwner
from func_lib import (
    default_model_fn,
    secure_mean,
    evaluate_classifier,
    secure_reptile,
    reptile_model_fn,
    mnist_data_pipeline
)

# Data flags
flags.DEFINE_string("remote_config", None, "Specify remote configuration")
flags.DEFINE_list("local_tfrecords", None,
                  ("Specify list of paths to TFRecord files on data owners "
                   "All paths supplied must exist on all data owners and "
                   "point to valid and identically processed TFRecord files. "
                   "If None, simulation will be assumed."))
flags.DEFINE_bool("validation_tfrecords", None,
                  ("Path to TFRecords on model owner to use for validation "
                   "data. If None, skip validation."))
flags.DEFINE_integer("evaluate_every", 20,
                     ("Evaluation frequency: the model owner will run an "
                      "evaluation step after this many rounds. If None, no "
                      "validation will happen."))
flags.DEFINE_integer("num_data_owners", 3,
                     ("Specify how many data owners should "
                      "take part in the learning. If remote "
                      "configuration is specified the number of data owners "
                      "there must match the number specified here."))

# Learning flags
flags.DEFINE_float("learning_rate", .01, "Global learning rate.")
flags.DEFINE_float("local_learning_rate", None,
                   ("Local learning rate (only used for reptile learning "
                    "rule). If None, defaults to `learning_rate`."))
flags.DEFINE_boolean("reptile", False,
                     ("If True, the ModelOwner will use the Reptile "
                      "meta-learning algorithm for performing updates to the "
                      "master model."))
flags.DEFINE_integer("rounds", 1000,
                     ("Number of rounds of training. A single round is "
                      "defined by one mini-batch update for each data owner, "
                      "plus an aggregation step by the model owner. A good "
                      "default would be: desired epochs * len(dataset) // "
                      "num_data_owners // batch_size."))
flags.DEFINE_integer("batch_size", 100, "Local batch size")

# Simulation flags
flags.DEFINE_boolean("simulation", True,
                     ("Whether the script runner should help simulate the "
                      "training by splitting the data and distributing it "
                      "among the participants."))
flags.DEFINE_string("dataset", "fashion_mnist:3.0.0",
                    ("Specify a TF Dataset to perform supervised learning on. "
                     "Must be a TF Dataset compatible with the S3 API. For "
                     "simulation only."))
flags.DEFINE_float("validation_split", .2,
                   ("Proportion of training set to hold out for validation on "
                    "model owner. For simulation only."))
flags.DEFINE_string("save_path", None,
                    ("Where the model should be saved. If None the "
                     "model isn't saved. With save_format 'h5' "
                     "it must be a path to a file, with 'tf' its a "
                     "path to a directory."))
flags.DEFINE_string("save_format", "h5",
                    ("Format of the saved model. Either 'h5' or 'tf',"
                     "'tf' saves it as a SavedModel."))

FLAGS = flags.FLAGS

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

  @from_simulated_dataset
  def build_data_pipeline(self):
    return mnist_data_pipeline(self.dataset, 50)


class DataOwner(BaseDataOwner):
  """Contains methods meant to be executed by a data owner.

  Args:
    player_name: `str`, name of the `tfe.player.Player`
                 representing the data owner
    build_update_step: `Callable`, the function used to construct
                       a local federated learning update.
  """

  @from_simulated_dataset
  def build_data_pipeline(self):
    return mnist_data_pipeline(self.dataset, FLAGS.batch_size)


def main(_):
  data_owners = [
      'data-owner-{}'.format(i) for i in range(FLAGS.num_data_owners)
  ]

  if FLAGS.remote_config is not None:
    config = tfe.RemoteConfig.load(FLAGS.remote_config)
    config.connect_to_cluster()
  else:
    players = ['server0', 'server1', 'crypto-producer', 'model-owner']
    config = tfe.EagerLocalConfig(players + data_owners)

  tfe.set_config(config)
  tfe.set_protocol(tfe.protocol.Pond())

  if FLAGS.simulation:
    if FLAGS.validation_split is None:
      model_owner = None
    else:
      model_owner = 'model-owner'

    simulation_tfrecords = federate_dataset(
        FLAGS.dataset,
        data_owners,
        model_owner,
        validation_split=FLAGS.validation_split,
    )

    dataowner_tfrecords = {do: simulation_tfrecords[do] for do in data_owners}
    validation_tfrecords = simulation_tfrecords.get('model-owner', None)
  else:
    simulation_tfrecords = None
    dataowner_tfrecords = FLAGS.local_tfrecords
    validation_tfrecords = FLAGS.validation_tfrecords



  logging.basicConfig(level=logging.DEBUG)

  model = tf.keras.Sequential((
      tf.keras.layers.Flatten(batch_input_shape=[FLAGS.batch_size, 28, 28, 1]),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dense(10),
  ))

  model.build()

  loss = tf.keras.losses.sparse_categorical_crossentropy

  local_lr = FLAGS.local_learning_rate or FLAGS.learning_rate

  model_owner = ModelOwner(
      # TODO: generalize this name?
      "model-owner",
      validation_tfrecords,
      model, loss,
      optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate)
  )

  data_owners = [DataOwner(data_owner,
                           dataowner_tfrecords[data_owner], model, loss,
                           optimizer=tf.keras.optimizers.Adam(local_lr))
                 for data_owner in dataowner_tfrecords]

  model_owner.fit(data_owners, rounds=FLAGS.rounds, evaluate_every=10)

  if FLAGS.save_path:
    model_owner.save_model(FLAGS.save_path, FLAGS.save_format)

if __name__ == "__main__":
  app.run(main)
