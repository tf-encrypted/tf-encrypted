import tensorflow as tf

# Compute global batch size using number of replicas.
BATCH_SIZE_PER_REPLICA = 5

def build_model():
  x = tf.keras.layers.Input(shape=(2,))
  y = tf.keras.layers.Dense(8)(x)
  y = tf.keras.layers.ReLU()(y)
  y = tf.keras.layers.Dense(1)(y)
  return tf.keras.models.Model(x, y)

def input_fn(global_batch_size):
  data = ({"feats": [[0., 0.], [1., 1.], [0., 1.], [1., 0.]]}, [[0.], [0.], [1.], [1.]])
  dataset = tf.data.Dataset.from_tensors(data)
  return dataset.repeat(100).batch(global_batch_size)

@tf.function
def train_step(strategy, dist_inputs):
  def step_fn(input_data):
    features, labels = input_data

    with tf.GradientTape() as tape:
      logits = model(features)
      cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
          logits=logits, labels=labels)
      loss = tf.reduce_sum(cross_entropy) * (1.0 / global_batch_size)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
    return cross_entropy

  per_example_losses = strategy.experimental_run_v2(
      step_fn, args=(dist_inputs,))
  mean_loss = strategy.reduce(
      tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)

  return tf.reduce_mean(mean_loss)

# print(help(type(tf.distribute.MirroredStrategy())))
mirrored_strategy = tf.distribute.MirroredStrategy()
global_batch_size = (BATCH_SIZE_PER_REPLICA *
                     mirrored_strategy.num_replicas_in_sync)

with mirrored_strategy.scope():
  model = build_model()
  optimizer = tf.keras.optimizers.SGD(learning_rate=.2)

  dataset = input_fn(global_batch_size)
  dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

  for epoch in range(15):
    for inputs in dist_dataset:
      print(train_step(mirrored_strategy, inputs))
