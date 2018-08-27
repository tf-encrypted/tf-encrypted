from typing import List
import sys

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe


config = tfe.LocalConfig([
    'server0',
    'server1',
    'crypto_producer',
    'model_owner',
    'prediction_client'
])

# config = tfe.RemoteConfig({
#     'server0': 'localhost:4440',
#     'server1': 'localhost:4441',
#     'crypto_producer': 'localhost:4442',
#     'model_owner': 'localhost:4443',
#     'prediction_client': 'localhost:4444'
# })


def load_training_data():
    mnist = tf.keras.datasets.mnist

    (x, y), _ = mnist.load_data()
    x = x / 255.0

    x = x.reshape((-1, 28*28))
    y = y.reshape((-1,))

    x = x.astype(np.float32)
    y = y.astype(np.int32)
    return x, y


def load_prediction_data():
    mnist = tf.keras.datasets.mnist

    (x, y), _ = mnist.load_data()
    # _, (x, y) = mnist.load_data()
    x = x / 255.0
    
    x = x.reshape((-1, 28*28))
    y = y.reshape((-1,))

    x = x[:10]
    y = y[:10]

    print("EXPECTED", y)

    x = x.astype(np.float32)
    return x


class ModelOwner(tfe.io.InputProvider):

    TRAINING_ITERATIONS = 50
    LEARNING_RATE = 0.01

    def provide_input(self) -> List[tf.Tensor]:

        # model parameters
        w0 = tf.Variable(tf.random_normal([28*28, 512]))
        b0 = tf.Variable(tf.zeros([512]))
        w1 = tf.Variable(tf.random_normal([512, 10]))
        b1 = tf.Variable(tf.zeros([10]))

        with tf.name_scope('training'):

            # load training data
            x, y = tf.py_func(load_training_data, [], [np.float32, np.int32])

            optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE)

            def loop_body(i):

                # model construction
                layer0 = tf.matmul(x, w0) + b0
                layer1 = tf.nn.sigmoid(layer0)
                layer2 = tf.matmul(layer1, w1) + b1

                predictions = layer2
                loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=predictions, labels=y))    
                with tf.control_dependencies([optimizer.minimize(loss)]):
                    return i + 1

            loop = tf.while_loop(lambda i: i < self.TRAINING_ITERATIONS, loop_body, (0,))

            with tf.control_dependencies([loop]):
                return [w0.read_value(), b0.read_value(), w1.read_value(), b1.read_value()]


class PredictionClient(tfe.io.InputProvider, tfe.io.OutputReceiver):

    def provide_input(self) -> List[tf.Tensor]:

        with tf.name_scope('data-loading'):
            x = tf.py_func(load_prediction_data, [], [np.float32])                
            x = tf.reshape(x, shape=(10, 28*28))

        return [x]

    def receive_output(self, tensors: List[tf.Tensor]) -> tf.Operation:
        likelihoods = tensors[0]
        prediction = tf.argmax(likelihoods, axis=1)
        return tf.Print([], [prediction], summarize=10, message="ACTUAL ")


if len(sys.argv) > 1:

    #
    # assume we're running as a server
    #

    player_name = str(sys.argv[1])

    # pylint: disable=E1101
    server = config.server(player_name)
    server.start()
    server.join()

else:

    #
    # assume we're running as master
    #

    model_owner = ModelOwner(config.get_player('model_owner'))
    prediction_client = PredictionClient(config.get_player('prediction_client'))

    with tfe.protocol.Pond(*config.get_players('server0, server1, crypto_producer')) as prot:

        # pylint: disable=E0632
        w0, b0, w1, b1 = prot.define_private_input(model_owner, masked=True)
        x, = prot.define_private_input(prediction_client, masked=True)

        # we'll use the same model weights several times
        w0, b0, w1, b1 = prot.cache([w0, b0, w1, b1])

        # compute prediction
        layer0 = prot.dot(x, w0) + b0
        layer1 = prot.sigmoid(layer0 * 0.1)
        layer2 = prot.dot(layer1, w1) + b1
        prediction = layer2

        # send output
        prediction_op = prot.define_output([prediction], prediction_client)

        with config.session() as sess:
            print("Init")
            tfe.run(sess, tf.global_variables_initializer(), tag='init')
            
            print("Training")
            tfe.run(sess, tfe.global_caches_updator(), tag='training')

            for _ in range(5):
                print("Predicting")
                tfe.run(sess, prediction_op, tag='prediction')
