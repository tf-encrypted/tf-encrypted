""" Base classes for the FL players """
#pylint:disable=unexpected-keyword-arg

import functools

import tensorflow as tf
import tf_encrypted as tfe
from tf_encrypted.protocol.pond import PondPrivateTensor

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

def pin_to_owner(func):

  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    with args[0].device:
      return func(*args, **kwargs)

    return pinned_fn

  return wrapper

class BaseModelOwner:
  """Contains code meant to be executed by some `ModelOwner` Player.

  Args:
    player_name: `str`, name of the `tfe.player.Player`
                 representing the model owner.
  """

  def __init__(self, player_name, local_data_file, model, loss, optimizer=None):
    self.player_name = player_name

    self.optimizer = optimizer
    self.loss = loss

    device_name = tfe.get_config().get_player(player_name).device_name
    self.device = tf.device(device_name)

    with self.device:
      # TODO: don't assume it's a tf.keras model
      self.model = tf.keras.models.clone_model(model)  # clone the model, get new weights

      self.evaluation_dataset = iter(build_data_pipeline(local_data_file, 50))

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
  def aggregator_fn(cls, model_gradients, model):
    raise NotImplementedError()

  @classmethod
  def evaluator_fn(cls, batches_or_owner):
    # TODO: figure out & fix the signature here
    raise NotImplementedError()

  def _runner(self, data_owners, rounds, evaluate, evaluate_every, **kwargs):
    """ Handles the train loop """
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
        # TODO: can we leak the loss here???
        loss = self._call_evaluator_fn()
        prog.update(r, [("Loss", loss)])

  @pin_to_owner
  def _call_aggregator_fn(self, grads, model):
    return self.aggregator_fn(grads, model)

  @pin_to_owner
  def _call_evaluator_fn(self):
    return self.evaluator_fn(self)

  @tfe.local_computation
  def call_model_fn(self, batches_or_owner):
    return self.model_fn(batches_or_owner)

  def _update_one_round(self, data_owners, **kwargs):
    """ Updates after one FL round """
    player_gradients = []

    # One round is an aggregation over all DataOwners
    for owner in data_owners:

      try:  # If the DataOwner implements a model_fn, use it
        grads = owner.call_model_fn(owner, player_name=owner.player_name,
                                    **kwargs)
        player_gradients.append(grads)
      except NotImplementedError:  # Otherwise, fall back to the ModelOwner's
        grads = self.call_model_fn(owner, player_name=owner.player_name,
                                   **kwargs)
        player_gradients.append(grads)

      # Fail gracefully
      except NotImplementedError: #pylint: disable=duplicate-except
        raise UndefinedModelFnError()

    # Aggregation step
    aggr_gradients = self._call_aggregator_fn(zip(*player_gradients),
                                              self.model)

    # Update step
    with self.device:
      if isinstance(aggr_gradients[0], PondPrivateTensor):
        aggr_gradients = [tf.cast(inp.reveal().to_native(), tf.float32)
                          for inp in aggr_gradients]

      self.optimizer.apply_gradients(zip(aggr_gradients,
                                         self.model.trainable_variables))


class BaseDataOwner:
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
    self.optimizer = optimizer

    device_name = tfe.get_config().get_player(player_name).device_name
    self.device = tf.device(device_name)

    with self.device:
      self.model = tf.keras.models.clone_model(model)
      self.dataset = iter(build_data_pipeline(local_data_file, 256))

  @tfe.local_computation
  def call_model_fn(self, batches_or_owner, *args, **kwargs):
    return self.model_fn(batches_or_owner, *args, **kwargs)

  @classmethod
  def model_fn(cls, batches_or_owner, *args, **kwargs):
    raise NotImplementedError()
