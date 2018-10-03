import tensorflow as tf

from data import gen_training_input, gen_test_input

tf.set_random_seed(1)

# Parameters
learning_rate = 0.01
training_set_size = 1000
test_set_size = 100
training_epochs = 10
batch_size = 100
nb_feats = 5


x, y = gen_training_input(training_set_size, nb_feats, batch_size)
x_test, y_test, _ = gen_test_input(test_set_size, nb_feats, batch_size)


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
# dc_dout = pred * (1 - pred) * dc_pred
# equivalent to:
dc_dout = pred - y
dW = tf.matmul(tf.transpose(x), dc_dout) / batch_size
db = tf.reduce_mean(1. * dc_dout, axis=0)
ops = [
    tf.assign(W, W - dW * learning_rate),
    tf.assign(b, b - db * learning_rate)
]


# Testing model
pred_test = tf.sigmoid(tf.matmul(x_test, W) + b)
correct_prediction = tf.equal(tf.round(pred_test), y_test)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Start training
total_batch = training_set_size // batch_size
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
