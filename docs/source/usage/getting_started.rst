Getting Started
================

This walkthrough assumes that you have installed `tf-encrypted` by following the `installation instructions`_.

.. _installation instructions: installation.html

`tf-encrypted` is a `secure multiparty computation`_ library where multiple people (or "`parties`") work together to compute results in a secure fashion without any one party having access to the underlying data. This is achieved by splitting up the input data into shares that are `perfectly secure`_.

.. _secure multiparty computation: https://en.wikipedia.org/wiki/Secure_multi-party_computation
.. _perfectly secure: https://en.wikipedia.org/wiki/One-time_pad

--------------------------------------------
Introduction to TensorFlow Encrypted API
--------------------------------------------

tf-encrypted provides an API similar to TensorFlow that data scientists and researchers can use to train models and predict upon them in privacy-preserving fashion.

One of the goals of `tf-encrypted` is to make experimenting with secure private machine learning accessible to anyone. To do this, we've implemented an API that is very similar to TensorFlow while abstracting away the complexity of securely managing public and private data. The `PondTensor` is the primary abstraction provided for managing public and private data.

The following example demonstrates constructing a public value (known to all parties) using `tfe.define_public_variable`.

.. code-block:: python

    import numpy as np
    import tf_encrypted as tfe

    variable = tfe.define_public_variable(np.array([1,2,3]))
    print(variable) # PondPublicVariable(shape=(3,))


We can then perform operations on these Tensors which define an underlying computation graph which can be executed inside a `Session`_ which manages figuring out which nodes run which parts of the computation. This is demonstrated in the following example:

.. _Session: ../api/session.html

.. code-block:: python

    variable = tfe.define_public_variable(np.array([1,2,3]))
    answer = variable * 2

    sess = tfe.Session()
    sess.run(tfe.global_variables_initializer(), tag='init') # ignore this for now :)
    sess.run(answer)

    # => array([2., 4., 6.])

Similar to public variables we can define private variables as demonstrated below:

.. code-block:: python

    variable = tfe.define_private_variable(np.array([1,2,3]))

    sess = tfe.Session()
    sess.run(tfe.global_variables_initializer(), tag='init')
    sess.run([variable.share0, variable.share1])

    # => [array([ 1601115100, -2072569751,  -600438257], dtype=int32),
    #     array([-1601049564,  2072700823,   600634865], dtype=int32)]

Unlike with public tensors, each node involved in a computation will get a different share of the encrypted (private) data. This sharing mechanism is the backbone of multiparty computation.

For more indepth examples of how to use `tf-encrypted` to train and predict upon machine learning models please check out our `MNIST`_ or `Logistic Regression`_ guies.

If you have any questions, please don't hesitate to reach out via a `GitHub Issue`_.

.. _`MNIST`: ../guides/mnist.html
.. _`Logistic Regression`: ../guides/logistic_regression.html
.. _`GitHub Issue`: https://github.com/mortendahl/tf-encrypted/issues

.. toctree::
    :maxdepth: 5
    :caption: Tutorials

    mnist
    logistic_regression
