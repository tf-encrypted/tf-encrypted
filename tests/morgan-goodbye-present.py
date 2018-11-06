import numpy as np
import tensorflow as tf
import tf_encrypted as tfe


tfe.set_protocol(tfe.protocol.SecureNN())

batch_size, channels_in, channels_out = 1, 1, 1
img_height, img_width = 5, 5
input_shape = (batch_size, channels_in, img_height, img_width)
input_conv = np.random.normal(size=input_shape).astype(np.float32)
x = tf.constant(input_conv, tf.float32)

# filters
h_filter, w_filter, strides = 2, 2, 2
filter_shape = (h_filter, w_filter, channels_in, channels_out)
filter_values = np.random.normal(size=filter_shape)
w = tf.constant(filter_values, tf.float32)

x_tf = tf.reshape(x, [-1, img_height, img_width, channels_in])
tf_out = tf.nn.relu(tf.nn.conv2d(x_tf, w, strides=[1, 1, 1, 1], padding='VALID'))
# tf_out = tf.nn.avg_pool(tf_out, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
# tf_out = tf.nn.conv2d(tf_out, w, strides=[1, 1, 1, 1], padding='VALID')
with tf.Session() as sess:
    print(sess.run(tf_out))

xp = tfe.define_private_input('input-provider', lambda: x)
wp = tfe.define_private_input('input-provider', lambda: w)
conv = tfe.conv2d(xp, wp, 1, 'VALID')
out = tfe.relu(conv)
# out = tfe.avgpool2d(out, (2, 2), (2, 2), 'VALID')
# out = tfe.conv2d(out, wp, 1, 'VALID')
with tfe.Session() as sess:
    res = sess.run([
        conv.reveal(),
        tfe.lsb(conv * 2).reveal(),
        out.reveal()
    ])
    for r in res:
        print(r)