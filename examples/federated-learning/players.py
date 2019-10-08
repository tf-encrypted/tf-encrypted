import functools

import tensorflow as tf
import tf_encrypted as tfe

from util import UndefinedModelFnError

from convert import decode

def build_data_pipeline(filename, batch_size):
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

class Owner(type):

  def __call__(self, *args, **kwargs):
    owner_obj = type.__call__(self, *args, **kwargs)

    # Decorate user-defined TF function that needs to be pinned to a particular
    # device and compiled
    if hasattr(owner_obj, "aggregator_fn"):
      self.handle_tf_local_fn(owner_obj, "aggregator_fn")

    if hasattr(owner_obj, "evaluator_fn"):
      self.handle_tf_local_fn(owner_obj, "evaluator_fn")

    # Decorate user-defined TFE local_computations
    if hasattr(owner_obj, "model_fn"):
      self.handle_tfe_local_fn(owner_obj, "model_fn")

    return owner_obj

  @classmethod
  def handle_tf_local_fn(mcs, owner_obj, func_name):
    func = getattr(owner_obj, func_name)
    pinned_evaluator = Owner.pin_to_owner(owner_obj)(func)
    setattr(owner_obj, func_name, pinned_evaluator)

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


class BaseModelOwner(metaclass=Owner):
  """Contains code meant to be executed by some `ModelOwner` Player.

  Args:
    player_name: `str`, name of the `tfe.player.Player`
                 representing the model owner.
  """

  def __init__(self, player_name, model, loss, optimizer=None):
    self.player_name = player_name

    self.optimizer = optimizer
    self.loss = loss

    device_name = tfe.get_config().get_player(player_name).device_name
    self.device = tf.device(device_name)

    with self.device:
      # TODO: don't assume it's a tf.keras model
      self.model = tf.keras.models.clone_model(model)  # clone the model, get new weights
      self.evaluation_dataset = iter(build_data_pipeline("./data/train.tfrecord", 50))

  def fit(self,
          data_owners,
          rounds,
          evaluate=True,
          evaluate_every=1,
          **model_fn_params):
    self._runner(
        data_owners, rounds, evaluate, evaluate_every, **model_fn_params,
    )

  ### User defined functions: see funclib.py for some examples ###
  @classmethod
  def model_fn(cls, batches_or_owner, *args, **kwargs):
    raise NotImplementedError()

  @classmethod
  def aggregator_fn(cls, weights_or_grads):
    raise NotImplementedError()

  @classmethod
  def evaluator_fn(cls, dataset):
    # TODO: figure out & fix the signature here
    raise NotImplementedError()

  def _runner(self, data_owners, rounds, evaluate, evaluate_every, **kwargs):
    prog = tf.keras.utils.Progbar(rounds, stateful_metrics=["Loss"])
    for r in range(rounds):
      # Train one step
      self._update_one_round(data_owners, **kwargs)

      # Broadcast master model
      for owner in data_owners:
        # TODO: don't assume it's a tf.keras model
        owner.model.set_weights(self.model.get_weights())

      # Evaluate once (maybe)
      if evaluate and (r + 1) % evaluate_every == 0:
        loss = self.evaluator_fn()
        prog.update(r, [("Loss", loss)])

  def _update_one_round(self, data_owners, **kwargs):
    player_gradients = []

    # One round is an aggregation over all DataOwners
    for owner in data_owners:

      try:  # If the DataOwner implements a model_fn, use it
        player_gradients.append(owner.model_fn(player_name=owner.player_name, **kwargs))

      except NotImplementedError:  # Otherwise, fall back to the ModelOwner's
        player_gradients.append(self.model_fn(owner, player_name=owner.player_name, **kwargs))

      except NotImplementedError:  # Fail gracefully
        raise UndefinedModelFnError()

    # Aggregation step
    aggr_gradients = self.aggregator_fn(zip(*player_gradients))

    # Update step
    with self.device:
      aggr_gradients = [tf.cast(aggr.reveal().to_native(), tf.float32)
                        for aggr in aggr_gradients]
      self.optimizer.apply_gradients(zip(aggr_gradients, self.model.trainable_variables))


class BaseDataOwner(metaclass=Owner):
  """Contains methods meant to be executed by a data owner.

  Args:
    player_name: `str`, name of the `tfe.player.Player`
                 representing the data owner
    build_update_step: `Callable`, the function used to construct
                       a local federated learning update.
  """
  def __init__(self, player_name, local_data_file, model, loss, optimizer=None):
    self.player_name = player_name
    self.local_data_file = local_data_file
    self.loss = loss

    device_name = tfe.get_config().get_player(player_name).device_name
    self.device = tf.device(device_name)

    with self.device:
      self.model = tf.keras.models.clone_model(model)
      self.dataset = iter(build_data_pipeline(local_data_file, 256))

  def model_fn(self, **kwargs):
    raise NotImplementedError()
