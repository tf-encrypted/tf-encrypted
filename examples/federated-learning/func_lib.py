import tensorflow as tf
import tf_encrypted as tfe


### Example model_fns ###

def default_model_fn(data_owner):
  """Runs a single training step!"""
  x, y = next(data_owner.dataset)

  with tf.name_scope("gradient_computation"):

    with tf.GradientTape() as tape:
      preds = data_owner.model(x)
      loss = tf.reduce_mean(data.loss(y, preds, from_logits=True))

    grads = tape.gradient(loss, data_owner.model.trainable_variables)

  return grads


def reptile_model_fn(data_owner, iterations=3, grad_fn=default_model_fn, **kwargs):
  # This corresponds to the Reptile variant that computes k steps of SGD.
  # When paired with the secure_aggregation aggregator_fn, this model_fn
  # corresponds to using g_k as the outer gradient update. See the Reptile
  # paper for more: https://arxiv.org/abs/1803.02999
  for k in range(iterations):
    grads_k = grad_fn(data_owner, **kwargs)
    data_owner.optimizer.apply_gradients(
        zip(grads_k, data_owner.model.trainable_variables),
    )

  return model.trainable_variables


### Example aggregator_fns ###

def secure_aggregation(collected_inputs):

  with tf.name_scope('secure_aggregation'):

    return [
        tfe.add_n(inputs) / len(inputs)
        for inputs in collected_inputs
    ]


### Example evaluator_fns ###

def evaluate_classifier(model_owner):
  """Runs a validation step!"""
  x, y = next(model_owner.dataset)

  with tf.name_scope('validate'):
    predictions = model_owner.model(x)
    loss = tf.reduce_mean(model_owner.loss(y, predictions, from_logits=True))

    y_hat = tf.argmax(input=predictions, axis=1)

  tf.print("loss", loss)
  tf.print("expect", y, summarize=50)
  tf.print("result", y_hat, summarize=50)
  return loss, y, y_hat
