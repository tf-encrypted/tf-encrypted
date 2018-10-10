from __future__ import absolute_import
import sys
import math
from typing import List

import tensorflow as tf
import tensorflow_encrypted as tfe

from examples.mnist.convert import decode


if len(sys.argv) >= 2:
    # config file was specified
    config_file = sys.argv[1]
    config = tfe.config.load(config_file)
else:
    # default to using local config
    config = tfe.LocalConfig([
        'server0',
        'server1',
        'crypto-producer',
        'model-trainer',
        'prediction-client'
    ])


class ModelTrainer():

    BATCH_SIZE = 32
    ITERATIONS = 60000 // BATCH_SIZE
    EPOCHS = 15
    IN_N = 28 * 28
    HIDDEN_N = 128
    OUT_N = 10

    def __init__(self, player: tfe.player.Player) -> None:
        self.player = player

    def build_data_pipeline(self):

        def normalize(image, label):
            x = tf.cast(image, tf.float32) / 255.
            image = (x - 0.1307) / 0.3081  # image = (x - mean) / std
            return image, label

        dataset = tf.data.TFRecordDataset(["./data/train.tfrecord"])
        dataset = dataset.map(decode)
        dataset = dataset.map(normalize)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.BATCH_SIZE)

        iterator = dataset.make_one_shot_iterator()
        return iterator

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
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

        # training loop
        def loop_body(i):

            # get next batch
            x, y = training_data.get_next()

            # model construction
            layer0 = x
            layer1 = tf.nn.relu(tf.matmul(layer0, w0) + b0)
            layer2 = tf.nn.relu(tf.matmul(layer1, w1) + b1)
            layer3 = tf.matmul(layer2, w2) + b2
            predictions = layer3

            loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=predictions, labels=y))
            with tf.control_dependencies([optimizer.minimize(loss)]):
                return i + 1

        loop = tf.while_loop(lambda i: i < self.ITERATIONS * self.EPOCHS, loop_body, (0,))

        # return model parameters after training
        loop = tf.Print(loop, [], message="Training complete")
        with tf.control_dependencies([loop]):
            return [param.read_value() for param in params]

    def provide_input(self) -> List[tf.Tensor]:
        with tf.name_scope('loading'):
            training_data = self.build_data_pipeline()

        with tf.name_scope('training'):
            parameters = self.build_training_graph(training_data)

        return parameters


class PredictionClient():

    BATCH_SIZE = 20

    def __init__(self, player: tfe.player.Player) -> None:
        self.player = player

    def build_data_pipeline(self):

        def normalize(image, label):
            x = tf.cast(image, tf.float32) / 255.
            image = (x - 0.1307) / 0.3081  # image = (x - mean) / std
            return image, label

        dataset = tf.data.TFRecordDataset(["./data/test.tfrecord"])
        dataset = dataset.map(decode)
        dataset = dataset.map(normalize)
        dataset = dataset.batch(self.BATCH_SIZE)

        iterator = dataset.make_one_shot_iterator()
        return iterator

    def provide_input(self) -> List[tf.Tensor]:
        with tf.name_scope('loading'):
            prediction_input, expected_result = self.build_data_pipeline().get_next()
            prediction_input = tf.Print(prediction_input, [expected_result], summarize=self.BATCH_SIZE, message="EXPECT ")

        with tf.name_scope('pre-processing'):
            prediction_input = tf.reshape(prediction_input, shape=(self.BATCH_SIZE, 28 * 28))

        return [prediction_input]

    def receive_output(self, likelihoods: tf.Tensor) -> tf.Operation:
        with tf.name_scope('post-processing'):
            prediction = tf.argmax(likelihoods, axis=1)
            op = tf.Print([], [prediction], summarize=self.BATCH_SIZE, message="ACTUAL ")
            return op


model_trainer = ModelTrainer(config.get_player('model-trainer'))
prediction_client = PredictionClient(config.get_player('prediction-client'))

server0 = config.get_player('server0')
server1 = config.get_player('server1')
crypto_producer = config.get_player('crypto-producer')

with tfe.protocol.Pond(server0, server1, crypto_producer) as prot:

    # get model parameters as private tensors from model owner
    params = prot.define_private_input(model_trainer.player, model_trainer.provide_input, masked=True)  # pylint: disable=E0632

    # we'll use the same parameters for each prediction so we cache them to avoid re-training each time
    params = prot.cache(params)

    # get prediction input from client
    x, = prot.define_private_input(prediction_client.player, prediction_client.provide_input, masked=True)  # pylint: disable=E0632

    # compute prediction
    w0, b0, w1, b1, w2, b2 = params
    layer0 = x
    layer1 = prot.relu((prot.matmul(layer0, w0) + b0))
    layer2 = prot.relu((prot.matmul(layer1, w1) + b1))
    logits = prot.matmul(layer2, w2) + b2

    # send prediction output back to client
    prediction_op = prot.define_output(prediction_client.player, [logits], prediction_client.receive_output)


with tfe.Session() as sess:
    print("Init")
    sess.run(tf.global_variables_initializer(), tag='init')

    print("Training")
    sess.run(tfe.global_caches_updator(), tag='training')

    for _ in range(5):
        print("Predicting")
        sess.run(prediction_op, tag='prediction')
