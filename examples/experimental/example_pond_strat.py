from . import tf_encrypted as tfe


strategy = tfe.protocol.Pond()

# x = provide_input(player="input-provider")
# y = provide_input(player="model-provider")
# assert x.dispatch_id == "private"
# assert y.dispatch_id == "private"

# Temporary:
x = tf.contant([1], dtype=tf.int64)
y = tf.constant([2], dtype=tf.int64)
x.dispatch_id = "private"
y.dispatch_id = "private"

with strategy.scope():
  z = tfe.add(x, y)
  z_revealed = strategy.reveal(z)

print(z_revealed)
