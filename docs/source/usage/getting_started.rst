Getting Started
================

This guide assumes that you have followed the `installation instructions`_.

.. _installation instructions: installation.html

There are also some concepts that you should be familiar with before reading this guide:

- `MPC`_: `tf-encrypted` is a secure multiparty computation library.  The short and suite is that this implies there will be multiple people (or "`parties`") working together to compute something in a secure fashion.  `i.e.` multiple people will get a result, but the computation and input data is `perfectly secure`_.

.. _MPC: https://en.wikipedia.org/wiki/Secure_multi-party_computation
.. _perfectly secure: https://en.wikipedia.org/wiki/One-time_pad

--------------------------------------------
Introduction to TensorFlow Encrypted API
--------------------------------------------

tf-encrypted has a simple api to make it easy for data scientists to make private predictions and trainings.
To define a machine learning model, tf-encrypted and TensorFlow follow a very similar API.
We can declare constants and variables in the same way that you are used to

.. code-block:: python

    import numpy as np
    import tf_encrypted as tfe

    variable = tfe.define_public_variable(np.array([1,2,3]))
    print(variable) # PondPublicVariable(shape=(3,))

One of the goals of the `tf-encrypted` API is to be able to abstract `MPC`_ details away from
users if they don't want to think about them.  In the above example we are able to abstract
one of the major challenges with implementing MPC algorithms, and that is data residency.

In MPC sometimes pieces of data are known to both parties, or sometimes only one person can know about a piece of data.
Obviously, private data should not leave the machine who knows what it really is, and public data is
okay to live anywhere, but it should be efficiently sent to machines that need it to avoid unnecessary communication.

The `PondTensor` above implements all of the appropriate abstractions to deal with these
challenges.  By default we will have 2 parties involved in a communication.  Creating
a tensor using `tfe.define_public_variable` ensures that the data ends up on the proper
machine.

.. _MPC: https://en.wikipedia.org/wiki/Secure_multi-party_computation

Expanding on this, we can start to see the benefits when we process the tensors

.. code-block:: python

    variable = tfe.define_public_variable(np.array([1,2,3]))
    answer = variable * 2

    sess = tfe.Session()
    sess.run(tfe.global_variables_initializer(), tag='init') # ignore this for now :)
    sess.run(answer)

    # => array([2., 4., 6.])

This hides the requirement of telling tensorflow to run this math on each individual machine.

Now, for something more interesting, private variables!

.. code-block:: python

    variable = tfe.define_private_variable(np.array([1,2,3]))

    sess = tfe.Session()
    sess.run(tfe.global_variables_initializer(), tag='init')
    sess.run([variable.share0, variable.share1])

    # => [array([ 1601115100, -2072569751,  -600438257], dtype=int32),
    #     array([-1601049564,  2072700823,   600634865], dtype=int32)]


Unlike the public tensor, private tensors will have different copies for the same
variable.  These `shares` are the backbone of MPC.

To get secrecy into your AI model though, you don't even have to worry about
this stuff.  Let's next see how we can run secret inputs into an AI model.

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Secret Logistic Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this section we will see how to do an easy task, but in secret: `Logistic Regression`_.

Let's go through piece by piece.  This section assumes some familiarity with machine learning and `TensorFlow`_.

.. _Logistic Regression: https://en.wikipedia.org/wiki/Logistic_regression
.. _TensorFlow: https://www.tensorflow.org/

.. code-block:: python

    import numpy as np
    import tensorflow as tf
    import tf_encrypted as tfe

    from data import gen_training_input, gen_test_input

    tf.set_random_seed(1)

    # Parameters
    learning_rate = 0.01
    training_set_size = 2000
    test_set_size = 100
    training_epochs = 10
    batch_size = 100
    nb_feats = 10

    xp, yp = tfe.define_private_input('input-provider', lambda: gen_training_input(training_set_size, nb_feats, batch_size))
    xp_test, yp_test = tfe.define_private_input('input-provider', lambda: gen_test_input(training_set_size, nb_feats, batch_size))

    W = tfe.define_private_variable(tf.random_uniform([nb_feats, 1], -0.01, 0.01))
    b = tfe.define_private_variable(tf.zeros([1]))


There is nothing here that should be too unfamiliar except the last four lines.

.. code-block:: python

    xp, yp = tfe.define_private_input('input-provider', lambda: gen_training_input(training_set_size, nb_feats, batch_size))
    xp_test, yp_test = tfe.define_private_input('input-provider', lambda: gen_test_input(training_set_size, nb_feats, batch_size))


.. TODO -- not super familiar about this wording

| This code creates two nodes in the tf graph that represent where private data & labels will enter the computation.
| See full code below of the `gen` methods.

.. code-block:: python

    W = tfe.define_private_variable(tf.random_uniform([nb_feats, 1], -0.01, 0.01))
    b = tfe.define_private_variable(tf.zeros([1]))


`W` and `b` represent the `weights` and `bias` of a classical neural network.  This network will train
the `weight` and `bias` to learn how to predict the generated sample data.

Next, we will declare how the model learns

.. code-block:: python

    out = tfe.matmul(xp, W) + b
    pred = tfe.sigmoid(out)

and the backprop

.. code-block:: python

    dc_dout = pred - yp
    dW = tfe.matmul(tfe.transpose(xp), dc_dout) * (1 / batch_size)
    db = tfe.reduce_sum(1. * dc_dout, axis=0) * (1 / batch_size)
    ops = [
        tfe.assign(W, W - dW * learning_rate),
        tfe.assign(b, b - db * learning_rate)
    ]

To test the model

.. code-block:: python

    pred_test = tfe.sigmoid(tfe.matmul(xp_test, W) + b)

Finally, we can run our training loop

.. code-block:: python

    def print_accuracy(pred_test_tf, y_test_tf: tf.Tensor) -> tf.Operation:
        correct_prediction = tf.equal(tf.round(pred_test_tf), y_test_tf)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return tf.Print(accuracy, data=[accuracy], message="Accuracy: ")


    print_acc_op = tfe.define_output('input-provider', [pred_test, yp_test], print_accuracy)

    total_batch = training_set_size // batch_size
    with tfe.Session() as sess:
        sess.run(tfe.global_variables_initializer(), tag='init')

        for epoch in range(training_epochs):
            avg_cost = 0.

            for i in range(total_batch):
                _, y_out, p_out = sess.run([ops, yp.reveal(), pred.reveal()], tag='optimize')
                # Our sigmoid function is an approximation
                # it can have values outside of the range [0, 1], we remove them and add/substract an epsilon to compute the cost
                p_out = p_out * (p_out > 0) + 0.001
                p_out = p_out * (p_out < 1) + (p_out >= 1) * 0.999
                c = -np.mean(y_out * np.log(p_out) + (1 - y_out) * np.log(1 - p_out))
                avg_cost += c / total_batch

            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")

        sess.run(print_acc_op)


You have just made a prediction without revealing anything about the input!
