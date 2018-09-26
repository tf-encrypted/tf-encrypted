import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe

from data import gen_training_input, gen_test_input


config = tfe.LocalConfig([
    'server0',
    'server1',
    'crypto_producer'
])
server0 = config.get_player('server0')
server1 = config.get_player('server1')
crypto_producer = config.get_player('crypto_producer')


# Parameters
learning_rate = 0.01
reg_rate = 10.  # Force weights to stay in [-1 ,1] range
training_set_size = 1000
test_set_size = 100
training_epochs = 5
batch_size = 100
nb_feats = 10


x, y = gen_training_input(training_set_size, nb_feats, batch_size)
x_test, y_test, y_np_out = gen_test_input(test_set_size, nb_feats, batch_size)


with tfe.protocol.Pond(server0, server1, crypto_producer) as prot:
    assert(isinstance(prot, tfe.protocol.Pond))

    W = prot.define_private_variable(tf.random_uniform([nb_feats, 1], -0.01, 0.01))
    b = prot.define_private_variable(tf.zeros([1]))
    xp = prot.define_private_input(x)  # We secure our inputs
    yp = prot.define_private_input(y)

    # Training model
    out = prot.matmul(xp, W) + b
    pred = prot.sigmoid(out)
    # l2_reg = .5 * (
    #     prot.reduce_sum(prot.square(W), axis=1) * (1/training_set_size) \
    #     + prot.reduce_sum(prot.square(b))
    # )
    # Due to missing log function approximation, we need to compute the cost in numpy
    # cost = -prot.reduce_sum(y * prot.log(pred) + (1 - y) * prot.log(1 - pred)) * (1/train_batch_size)

    # Backprop
    dc_out = pred - yp
    dW = prot.matmul(prot.transpose(xp), dc_out)
    db = prot.reduce_sum(1. * dc_out, axis=0)
    ops = [
        prot.assign(W, W - dW * learning_rate),
        prot.assign(b, b - db * learning_rate)
    ]

    # Testing model
    xp_test = prot.define_private_input(x_test)  # We secure our inputs
    yp_test = prot.define_private_input(y_test)
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
