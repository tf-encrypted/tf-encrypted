from typing import Tuple

import numpy as np
import tensorflow as tf


# Parameters
learning_rate = 0.001
training_epochs = 10
train_batch_size = 100
test_batch_size = 100
nb_feats = 5


def norm(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.cast(x, tf.float32), tf.expand_dims(y, 0)


x_np = np.random.uniform(-1 / 2, 1 / 2, size=[10000, nb_feats])
y_np = np.array(x_np.mean(axis=1) > 0, np.float32)
train_set = tf.data.Dataset.from_tensor_slices((x_np, y_np)) \
                           .map(norm) \
                           .shuffle(buffer_size=100) \
                           .repeat() \
                           .batch(train_batch_size)
train_set_iterator = train_set.make_one_shot_iterator()
x, y = train_set_iterator.get_next()
x = tf.reshape(x, [train_batch_size, nb_feats])
y = tf.reshape(y, [train_batch_size, 1])


x_test_np = np.random.uniform(-1 / 2, 1 / 2, size=[100, nb_feats])
y_test_np = np.array(x_test_np.mean(axis=1) > 0, np.float32)
test_set = tf.data.Dataset.from_tensor_slices((x_test_np, y_test_np)) \
                          .map(norm) \
                          .batch(test_batch_size)
test_set_iterator = test_set.make_one_shot_iterator()
x_test, y_test = test_set_iterator.get_next()
x_test = tf.reshape(x_test, [train_batch_size, nb_feats])
y_test = tf.reshape(y_test, [train_batch_size, 1])


W = tf.Variable(tf.random_uniform([nb_feats, 1], -0.01, 0.01))
b = tf.Variable(tf.zeros([1]))


# Training model
out = tf.matmul(x, W) + b
pred = tf.sigmoid(out)
cost = -tf.reduce_mean(y * tf.log(pred) + (1 - y) * tf.log(1 - pred))


# Backprop
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# equivalent to:
# dc_pred = - y / pred + (1 - y) / (1 - pred)
# dc_out = pred * (1 - pred) * dc_pred
# equivalent to:
dc_out = pred - y
dW = tf.matmul(tf.transpose(x), dc_out)
db = tf.reduce_sum(1. * dc_out, axis=0)
ops = [
    tf.assign(W, W - dW * learning_rate),
    tf.assign(b, b - db * learning_rate)
]


# Testing model
pred_test = tf.sigmoid(tf.matmul(x_test, W) + b)
correct_prediction = tf.equal(tf.round(pred_test), y_test)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Start training
total_batch = int(len(x_np) / train_batch_size)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.

        for i in range(total_batch):
            _, c = sess.run([ops, cost])
            avg_cost += c / total_batch

        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    print("Accuracy:", accuracy.eval())
