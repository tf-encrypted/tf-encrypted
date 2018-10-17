Getting Started
================

This guide assumes that you have followed the `installation instructions`_.

.. _installation instructions: installation.html

--------------------------------------------
Introduction to TensorFlow Encrypted API
--------------------------------------------

tf-encrypted has a simple api to make it easy for data scientists to make private predictions and trainings.


To define a machine learning model, tf-encrypted and TensorFlow follow a very similar API.

^^^^^^^^^^^
TensorFlow
^^^^^^^^^^^

.. code-block:: python

    layer0 = tf.matmul(x, w0) + b0
    layer1 = tf.nn.sigmoid(layer0)
    layer2 = tf.matmul(layer, w1) + b1
    prediction = layer2


^^^^^^^^^^^^^
tf-encrypted
^^^^^^^^^^^^^

.. code-block:: python

    layer0 = prot.dot(x, w0) + b0
    layer1 = prot.sigmoid(layer0)
    layer2 = prot.dot(layer, w1) + b1
    prediction = layer2


To make private predictions and trainings, we have to define a ModelTrainer and a
PredictionClient. The ModelTrainer is responsible for training the model,
and then provides the encrypted weights. The PredictionClient will provide the
private input that will be used to make a private prediction.

.. code-block:: python

    model_trainer = ModelTrainer(config.get_player('model-trainer'))
    prediction_client = PredictionClient(config.get_player('prediction-client'))


We indicate tf-encrypted to perform secure computation with the Pond protocol as follows

.. code-block:: python

    with tfe.protocol.Pond(server0, server1, crypto_producer) as prot:


Finally we can make a prediction as like so

.. code-block:: python

    with tfe.protocol.Pond(server0, server1, crypto_producer) as prot:

      # get model parameters as private tensors from model owner
      w0, b0, w1, b1 = prot.define_private_input(model_trainer, masked=True)

      # get prediction input from client
      x = prot.define_private_input(prediction_client, masked=True)

      # compute prediction
      layer0 = prot.dot(x, w0) + b0
      layer1 = prot.sigmoid(layer0)
      layer2 = prot.dot(layer, w1) + b1
      prediction = layer2

      # send prediction output back to client
      prediction_op = prot.define_output(prediction, prediction_client)


Two servers and a crypto producer are doing the actual computation on encrypted data,
with only the client being able to decrypt the final result.


---------------------------
Making Private Predictions
---------------------------

^^^^^^^^^^^
a. Locally
^^^^^^^^^^^

With Tensorflow Encrypted, you can very easily make private predictions with a
pre-trained model saved as a `protobuf`_. file.

1. In the `examples/`_ folder run the following line of code in your terminal to train a convolution network on MNIST dataset, then save the model as a protobuf file :

.. code-block:: bash

    python3 mnist_deep_cnn.py

If you prefer to skip this step, we have saved the trained model `mnist_model.pb`_.

2. Make private prediction on an MNIST input by running the following code snippet:

.. code-block:: bash

    ../bin/run test_data/mnist_model.pb test_data/mnist_input.npy


.. _protobuf: https://www.tensorflow.org/extend/tool_developers/
.. _examples/: https://github.com/mortendahl/tf-encrypted/tree/master/examples
.. _mnist_model.pb: https://github.com/mortendahl/tf-encrypted/tree/master/examples/test_data


You have just made a prediction without revealing anything about the input!
