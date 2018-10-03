import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe

from data import gen_training_input, gen_test_input

tfe.set_random_seed(1)

# Parameters
learning_rate = 0.01
training_set_size = 1000
test_set_size = 100
training_epochs = 10
batch_size = 100
nb_feats = 5

with tfe.protocol.Pond() as prot:
    assert(isinstance(prot, tfe.protocol.Pond))

    x, y = gen_training_input(training_set_size, nb_feats, batch_size)
    x_test, y_test, y_np_out = gen_test_input(test_set_size, nb_feats, batch_size)

    xp = prot.define_private_input(
        tfe.io.InputProvider('input-provider', lambda: gen_training_input(training_set_size, nb_feats, batch_size)[0])
    )
    yp = prot.define_private_input(
        tfe.io.InputProvider('input-provider', lambda: gen_training_input(training_set_size, nb_feats, batch_size)[1])
    )

    W = prot.define_private_variable(tf.random_uniform([nb_feats, 1], -0.01, 0.01))
    b = prot.define_private_variable(tf.zeros([1]))

    # Training model
    out = prot.matmul(xp, W) + b
    pred = prot.sigmoid(out)
    # Due to missing log function approximation, we need to compute the cost in numpy
    # cost = -prot.sum(y * prot.log(pred) + (1 - y) * prot.log(1 - pred)) * (1/train_batch_size)

    # Backprop
    dc_dout = pred - yp
    dW = prot.matmul(prot.transpose(xp), dc_dout) * (1 / batch_size)
    db = prot.reduce_sum(1. * dc_dout, axis=0) * (1 / batch_size)
    ops = [
        prot.assign(W, W - dW * learning_rate),
        prot.assign(b, b - db * learning_rate)
    ]

    # Testing model
    xp_test = prot.define_private_input(
        tfe.io.InputProvider('input-provider', lambda: gen_test_input(training_set_size, nb_feats, batch_size)[0])
    )
    yp_test = prot.define_private_input(
        tfe.io.InputProvider('input-provider', lambda: gen_test_input(training_set_size, nb_feats, batch_size)[1])
    )
    pred_test = prot.sigmoid(prot.matmul(xp_test, W) + b)

    total_batch = training_set_size // batch_size
    with tfe.Session() as sess:
        sess.run(prot.initializer, tag='init')

        for epoch in range(training_epochs):
            avg_cost = 0.

            for i in range(total_batch):
                _, y_out, p_out = sess.run([ops, y, pred.reveal()], tag='optimize')
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
