import tensorflow as tf
import tf_encrypted as tfe

shape = (10, 10, 3)
x = tf.keras.layers.Input(shape=shape)
y = tf.keras.layers.Conv2D(16, (3, 3))(x)
y = tf.keras.layers.MaxPooling2D((2, 2), (2, 2))(y)
y = tf.keras.layers.Flatten()(y)
y = tf.keras.layers.Dense(10)(y)

model = tf.keras.Model(inputs=x, outputs=y)

# Helper to inspect the incoming graph, to ensure that TFE has conversion
# functions for everything you're requesting.
sess = tf.keras.backend.get_session()
tfe.convert.inspect_subgraph(model, shape, sess)

# Idiomatic way of converting in a specific protocol
with tfe.protocol.SecureNN():
  s_model = tfe.private_model.secure_model(model)

# This one should work as well
prot = tfe.protocol.SecureNN()
s_model = tfe.private_model.secure_model(model, protocol=prot)
