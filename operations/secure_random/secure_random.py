import tensorflow as tf
secure_random_module = tf.load_op_library('./secure_random.so')
with tf.Session(''):
    print(secure_random_module.secure_random([5, 6], [1, 1, 1, 1, 1, 1, 1, 1]).eval())
