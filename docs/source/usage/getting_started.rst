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
this stuff.  Check the other pages to see how you can start using MPC with machine learning.

.. toctree::
    :maxdepth: 5
    :caption: Tutorials

    mnist
    logistic_regression
