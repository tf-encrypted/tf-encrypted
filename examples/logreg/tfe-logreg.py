from typing import Union
import numpy as np

import tensorflow as tf
import tensorflow_encrypted as tfe
from tensorflow_encrypted.protocol.pond import PondPrivateTensor

from data import gen_training_input, gen_test_input


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
learning_rate = 0.01
training_set_size = 1000
test_set_size = 100
training_epochs = 10
batch_size = 100
nb_feats = 5


x, y = gen_training_input(training_set_size, nb_feats, batch_size)
x_test, y_test, y_np_out = gen_test_input(test_set_size, nb_feats, batch_size)


with tfe.protocol.Pond(server0, server1, crypto_producer) as prot:
    assert(isinstance(prot, tfe.protocol.Pond))

    W = prot.define_private_variable(tf.random_uniform([nb_feats, 1], -0.01, 0.01))
    b = prot.define_private_variable(tf.zeros([1]))
    xp = consume(x, prot)  # We secure our inputs
    yp = consume(y, prot)

    # Training model
    out = prot.dot(xp, W) + b
    pred = prot.sigmoid(out)
    # Due to missing log function approximation, we need to compute the cost in numpy
    # cost = -prot.sum(y * prot.log(pred) + (1 - y) * prot.log(1 - pred)) * (1/train_batch_size)

    # Backprop
    dc_out = pred - yp
    dW = prot.dot(prot.transpose(xp), dc_out)
    db = prot.sum(1. * dc_out, axis=0, keepdims=False)
    ops = [
        prot.assign(W, W - dW * learning_rate),
        prot.assign(b, b - db * learning_rate)
    ]

    # Testing model
    xp_test = consume(x_test, prot)  # We secure our inputs
    yp_test = consume(y_test, prot)
    pred_test = prot.sigmoid(prot.dot(xp_test, W) + b)

    total_batch = training_set_size // batch_size
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

            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")

        out_test = pred_test.reveal().eval(sess)
        acc = np.mean(np.round(out_test) == y_np_out)
        print("Accuracy:", acc)
