from typing import Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe
from tensorflow_encrypted.protocol.pond import PondPrivateTensor

config = tfe.LocalConfig([
    'server0',
    'server1',
    'crypto_producer'
])
server0 = config.get_player('server0')
server1 = config.get_player('server1')
crypto_producer = config.get_player('crypto_producer')

def consume(v: Union[np.ndarray, tf.Tensor], prot: tfe.protocol.Pond) -> PondPrivateTensor:
    val = tfe.protocol.pond._encode(v, True)
    x0, x1 = tfe.protocol.pond._share(val)
    x = PondPrivateTensor(prot, x0, x1, True)
    return x

# Parameters
learning_rate = 0.001
training_epochs = 10
train_batch_size = 100
test_batch_size = 100
nb_feats = 5

def norm(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.cast(x, tf.float32), tf.expand_dims(y, 0)

x_np = np.random.uniform(-1/2, 1/2, size=[10000, nb_feats])
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

x_test_np = np.random.uniform(-1/2, 1/2, size=[100, nb_feats])
y_test_np = np.array(x_test_np.mean(axis=1) > 0, np.float32)
test_set = tf.data.Dataset.from_tensor_slices((x_test_np, y_test_np)) \
                          .map(norm) \
                          .batch(test_batch_size)
test_set_iterator = test_set.make_one_shot_iterator()
x_test, y_test = test_set_iterator.get_next()
x_test = tf.reshape(x_test, [train_batch_size, nb_feats])
y_test = tf.reshape(y_test, [train_batch_size, 1])

with tfe.protocol.Pond(server0, server1, crypto_producer) as prot:
    assert(isinstance(prot, tfe.protocol.Pond))

    W = prot.define_private_variable(tf.random_uniform([nb_feats, 1], -0.01, 0.01))
    b = prot.define_private_variable(tf.zeros([1]))
    xp = consume(x, prot) # We secure our inputs
    yp = consume(y, prot)

    # Training model
    out = prot.dot(xp, W) + b
    pred = prot.sigmoid(out)
    # Due to missing log function approximation, we need to compute the cost in numpy
    # cost = -prot.sum(y * prot.log(pred) + (1 - y) * prot.log(1 - pred)) * (1/train_batch_size)

    # Backprop
    dout = pred - yp
    dW = prot.dot(prot.transpose(xp), dout) # [nb_feats, 1] <- [nb_feats, bs].[bs, 1]
    db = prot.sum(dout * 1.0, axis=0, keepdims=False)
    ops = [
        prot.assign(W, W - dW * learning_rate),
        prot.assign(b, b - db * learning_rate)
    ]

    # Testing model
    xp_test = consume(x_test, prot) # We secure our inputs
    yp_test = consume(y_test, prot)
    pred_test = prot.sigmoid(prot.dot(xp_test, W) + b)

    total_batch = int(len(x_np) / train_batch_size)
    with config.session() as sess:
        tfe.run(sess, prot.initializer, tag='init')

        for epoch in range(training_epochs):
            avg_cost = 0.

            for i in range(total_batch):
                p_out = pred.reveal().eval(sess)
                _, y_out = tfe.run(sess, [ops, y], tag='optimize')
                # Our sigmoid function is an approximation
                # it can have values outside of the range [0, 1], we remove them and add/substract an epsilon to compute the cost
                p_out = p_out * (p_out > 0) + 0.001
                p_out = p_out * (p_out < 1) + (p_out >= 1) * 0.999
                c = -np.mean(y_out * np.log(p_out) + (1 - y_out) * np.log(1 - p_out))
                avg_cost += c / total_batch

            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")

        out_test = pred_test.reveal().eval(sess)
        acc = np.mean(np.round(out_test) == y_test_np)
        print("Accuracy:", acc)
