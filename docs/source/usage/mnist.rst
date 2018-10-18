MNIST
======

This tutorial is also available on `Google Collab`_, feel free to follow along there!

In this tutorial, we will train our model in plaintext with Tensorflow, then
make private predictions with `tf-encrypted`. we will use the `MNIST dataset`_.

.. _Google Collab: https://colab.research.google.com/drive/1fc7FYSWE2c5s_LsRDAlTPZB_TSAImtYq?authuser=2&pli=1#scrollTo=-QBbU7bBr39p
.. _MNIST dataset: http://yann.lecun.com/exdb/mnist/

.. code-block:: python

    from __future__ import absolute_import
    import os
    import sys
    import math
    from typing import List, Tuple

    import tensorflow as tf
    import tf_encrypted as tfe

    from tensorflow.keras.datasets import mnist

We save the MNIST data in TFRecord format which is the recommended format for TensorFlow.
Below are just helper functions to encode and decode the images and the labels in the right format.
To build the input pipeline, we use `tf.data.TFRecordDataset`.
This object is very handy if we want to chain operations such as normalizing the inputs, generating batches etc.

.. code-block:: python

    def encode_image(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tostring()]))


    def decode_image(value):
        image = tf.decode_raw(value, tf.uint8)
        image.set_shape((28 * 28))
        return image


    def encode_label(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    def decode_label(value):
        return tf.cast(value, tf.int32)


    def encode(image, label):
        return tf.train.Example(features=tf.train.Features(feature={
            'image': encode_image(image),
            'label': encode_label(label)
        }))


    def decode(serialized_example):
        features = tf.parse_single_example(serialized_example, features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        })
        image = decode_image(features['image'])
        label = decode_label(features['label'])
        return image, label


    def normalize(image, label):
        x = tf.cast(image, tf.float32) / 255.
        image = (x - 0.1307) / 0.3081  # image = (x - mean) / std
        return image, label


    def get_data_from_tfrecord(filename: str, bs: int) -> Tuple[tf.Tensor, tf.Tensor]:
        return tf.data.TFRecordDataset([filename]) \
                      .map(decode) \
                      .map(normalize) \
                      .repeat() \
                      .batch(bs) \
                      .make_one_shot_iterator()

    def save_training_data(images, labels, filename):
        assert images.shape[0] == labels.shape[0]
        num_examples = images.shape[0]

        with tf.python_io.TFRecordWriter(filename) as writer:

            for index in range(num_examples):

                image = images[index]
                label = labels[index]
                example = encode(image, label)
                writer.write(example.SerializeToString())


    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    data_dir = os.path.expanduser("./data/")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    save_training_data(x_train, y_train, os.path.join(data_dir, "train.tfrecord"))
    save_training_data(x_test, y_test, os.path.join(data_dir, "test.tfrecord"))

Below is just an helper function to print tensors in a notebook.

.. code-block:: python

    # Source: https://stackoverflow.com/questions/37898478/is-there-a-way-to-get-tensorflow-tf-print-output-to-appear-in-jupyter-notebook-o
    def tf_print(tensor, transform=None):

    def print_tensor(x):
        print(x if transform is None else transform(x))
        return x
    log_op = tf.py_func(print_tensor, [tensor], [tensor.dtype])[0]
    with tf.control_dependencies([log_op]):
        res = tf.identity(tensor)

    return res

----------------------------------
Select your cryptography protocol
----------------------------------

In this example we use the SecureNN protocol. As for the different parties involved,
we here assume a setting with two server, a crypto producer,
a weights provider (model-trainer), and a private input provider (prediction-client).
Note that we could have selected very easily the Pond protocol by running instead:
`tfe.set_protocol(tfe.protocol.Pond(*tfe.get_config().get_players(['server0', 'server1', 'crypto-producer'])))`

.. code-block:: python

    config = tfe.LocalConfig([
            'server0',
            'server1',
            'crypto-producer',
            'model-trainer',
            'prediction-client'
        ])


    tfe.set_config(config)
    tfe.set_protocol(tfe.protocol.SecureNN(*tfe.get_config().get_players(['server0', 'server1', 'crypto-producer'])))

-------------------
Plaintext Training
-------------------

Then we create a `ModelTrainer` object which is responsible for training the model
in plaintext then provides the weights to perform private predictions.

.. code-block:: python

    class ModelTrainer():

        BATCH_SIZE = 256
        ITERATIONS = 60000 // BATCH_SIZE
        EPOCHS = 3
        LEARNING_RATE = 3e-3
        IN_N = 28 * 28
        HIDDEN_N = 128
        OUT_N = 10

        def cond(self, i: tf.Tensor, max_iter: tf.Tensor, nb_epochs: tf.Tensor, avg_loss: tf.Tensor) -> tf.Tensor:
            is_end_epoch = tf.equal(i % max_iter, 0)
            to_continue = tf.cast(i < max_iter * nb_epochs, tf.bool)

            def true_fn() -> tf.Tensor:
                #tf_print(tensor, transform=None)
                #res = tf_print(avg_loss)
                #return res
                return tf.Print(to_continue, data=[avg_loss], message="avg_loss: ")

            def false_fn() -> tf.Tensor:
                return to_continue

            return tf.cond(is_end_epoch, true_fn, false_fn)

        def build_training_graph(self, training_data) -> List[tf.Tensor]:
            j = self.IN_N
            k = self.HIDDEN_N
            m = self.OUT_N
            r_in = math.sqrt(12 / (j + k))
            r_hid = math.sqrt(12 / (2 * k))
            r_out = math.sqrt(12 / (k + m))

            # model parameters and initial values
            w0 = tf.Variable(tf.random_uniform([j, k], minval=-r_in, maxval=r_in))
            b0 = tf.Variable(tf.zeros([k]))
            w1 = tf.Variable(tf.random_uniform([k, k], minval=-r_hid, maxval=r_hid))
            b1 = tf.Variable(tf.zeros([k]))
            w2 = tf.Variable(tf.random_uniform([k, m], minval=-r_out, maxval=r_out))
            b2 = tf.Variable(tf.zeros([m]))
            params = [w0, b0, w1, b1, w2, b2]

            # optimizer and data pipeline
            optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE)

            # training loop
            def loop_body(i: tf.Tensor, max_iter: tf.Tensor, nb_epochs: tf.Tensor, avg_loss: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
                # get next batch
                x, y = training_data.get_next()

                # model construction
                layer0 = x
                layer1 = tf.nn.relu(tf.matmul(layer0, w0) + b0)
                layer2 = tf.nn.relu(tf.matmul(layer1, w1) + b1)
                predictions = tf.matmul(layer2, w2) + b2

                loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=predictions, labels=y))

                is_end_epoch = tf.equal(i % max_iter, 0)

                def true_fn() -> tf.Tensor:
                    return loss

                def false_fn() -> tf.Tensor:
                    return (tf.cast(i - 1, tf.float32) * avg_loss + loss) / tf.cast(i, tf.float32)

                with tf.control_dependencies([optimizer.minimize(loss)]):
                    return i + 1, max_iter, nb_epochs, tf.cond(is_end_epoch, true_fn, false_fn)

            loop, _, _, _ = tf.while_loop(self.cond, loop_body, [0, self.ITERATIONS, self.EPOCHS, 0.])

            # return model parameters after training
            loop = tf.Print(loop, [], message="Training complete")
            with tf.control_dependencies([loop]):
                return [param.read_value() for param in params]

        def provide_input(self) -> List[tf.Tensor]:
            with tf.name_scope('loading'):
                training_data = get_data_from_tfrecord("./data/train.tfrecord", self.BATCH_SIZE)

            with tf.name_scope('training'):
                parameters = self.build_training_graph(training_data)

            return parameters

--------------------
Private Predictions
--------------------

The `PredictionClient` object will provide the private input that will be used to make a private prediction.

.. code-block:: python

    class PredictionClient():

    BATCH_SIZE = 20

    def provide_input(self) -> List[tf.Tensor]:
        with tf.name_scope('loading'):
            prediction_input, expected_result = get_data_from_tfrecord("./data/test.tfrecord", self.BATCH_SIZE).get_next()

        with tf.name_scope('pre-processing'):
            prediction_input = tf.reshape(prediction_input, shape=(self.BATCH_SIZE, 28 * 28))
            expected_result = tf.reshape(expected_result, shape=(self.BATCH_SIZE,))

        return [prediction_input, expected_result]

    def receive_output(self, likelihoods: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('post-processing'):
            prediction = tf.argmax(likelihoods, axis=1)
            eq_values = tf.equal(prediction, tf.cast(y_true, tf.int64))
            acc = tf.reduce_mean(tf.cast(eq_values, tf.float32))
            op = tf.Print([], [y_true], summarize=self.BATCH_SIZE, message="EXPECT: ")
            op = tf.Print(op, [prediction], summarize=self.BATCH_SIZE, message="ACTUAL: ")
            op = tf_print(prediction)
            op = tf.Print([op], [acc], summarize=self.BATCH_SIZE, message="Acuraccy: ")
            return op


Once you instantiate the `ModelTrainer` and `PredictionClient` objects, you can very
easily get the weights trained in plaintext, get the private input from the client
and finally make private predictions. As you can see, to create a model, `tf-encrypted`
and TensorFlow follow a very similar API

.. code-block:: python

    layer0 = x
    layer1 = tfe.relu((tfe.matmul(layer0, w0) + b0))
    layer2 = tfe.relu((tfe.matmul(layer1, w1) + b1))
    logits = tfe.matmul(layer2, w2) + b2


.. code-block:: python

    model_trainer = ModelTrainer()
    prediction_client = PredictionClient()

    # get model parameters as private tensors from model owner
    params = tfe.define_private_input('model-trainer', model_trainer.provide_input, masked=True)

    # we'll use the same parameters for each prediction so we cache them to avoid re-training each time
    params = tfe.cache(params)

    # get prediction input from client
    x, y = tfe.define_private_input('prediction-client', prediction_client.provide_input, masked=True)

    # compute prediction
    w0, b0, w1, b1, w2, b2 = params
    layer0 = x
    layer1 = tfe.relu((tfe.matmul(layer0, w0) + b0))
    layer2 = tfe.relu((tfe.matmul(layer1, w1) + b1))
    logits = tfe.matmul(layer2, w2) + b2

    # send prediction output back to client
    prediction_op = tfe.define_output('prediction-client', [logits, y], prediction_client.receive_output)

    with tfe.Session() as sess:
        print("Init")
        sess.run(tf.global_variables_initializer(), tag='init')

        print("Training")
        sess.run(tfe.global_caches_updator(), tag='training')

        for _ in range(5):
            print("Private Predictions:")
            sess.run(prediction_op, tag='prediction')


And voila! you have just trained a model in plaintext then made private predictions
without revealing anything about the input!
