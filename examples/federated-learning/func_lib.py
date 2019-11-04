""" Contains example functions for handling training """

import tensorflow as tf
import tf_encrypted as tfe

from convert import decode

### Example model_fns ###

def default_model_fn(data_owner):
  """Runs a single training step!"""
  x, y = next(data_owner.dataset)

  with tf.name_scope("gradient_computation"):

    with tf.GradientTape() as tape:
      preds = data_owner.model(x)
      loss = tf.reduce_mean(data_owner.loss(y, preds, from_logits=True))

    grads = tape.gradient(loss, data_owner.model.trainable_variables)

  return grads

def reptile_model_fn(data_owner, iterations=3,
                     grad_fn=default_model_fn, **kwargs):
  """
  This corresponds to the Reptile variant that computes k steps of SGD.
  When paired with the secure_aggregation aggregator_fn, this model_fn
  corresponds to using g_k as the outer gradient update. See the Reptile
  paper for more: https://arxiv.org/abs/1803.02999
  """

  for _ in range(iterations):
    grads_k = grad_fn(data_owner, **kwargs)
    data_owner.optimizer.apply_gradients(
        zip(grads_k, data_owner.model.trainable_variables),
    )

  return [var.read_value() for var in data_owner.model.trainable_variables]


### Example aggregator_fns ###

def secure_mean(collected_inputs):
  """ securely calculates the mean of the collected_inputs """

  with tf.name_scope('secure_mean'):

    aggr_inputs = [
        tfe.add_n(inputs) / len(inputs)
        for inputs in collected_inputs
    ]

    # Reveal aggregated values & cast to native tf.float32
    aggr_inputs = [tf.cast(inp.reveal().to_native(), tf.float32)
                   for inp in aggr_inputs]

    return aggr_inputs

def secure_reptile(collected_inputs, model):

  aggr_weights = secure_mean(collected_inputs)

  weights_deltas = [
      weight - update for (weight, update) in zip(
          model.trainable_variables, aggr_weights,
      )
  ]
  return weights_deltas


### Example evaluator_fns ###

def evaluate_classifier(model_owner):
  """Runs a validation step!"""
  x, y = next(model_owner.evaluation_dataset)

  with tf.name_scope('validate'):
    predictions = model_owner.model(x)
    loss = tf.reduce_mean(model_owner.loss(y, predictions, from_logits=True))

  return loss

def mnist_data_pipeline(dataset, batch_size):
  """Build data pipeline for validation by model owner."""
  def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

  dataset = dataset.map(decode)
  dataset = dataset.map(normalize)
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.repeat()

  return dataset
