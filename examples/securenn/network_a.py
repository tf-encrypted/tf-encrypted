from __future__ import absolute_import
import sys
import math
from typing import List, Tuple

import tensorflow as tf
import tf_encrypted as tfe

from examples.mnist.convert import get_data_from_tfrecord

# tfe.setMonitorStatsFlag(True)

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
tfe.set_config(config)
tfe.set_protocol(tfe.protocol.SecureNN(*tfe.get_config().get_players(['server0', 'server1', 'crypto-producer'])))


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
            op = tf.Print([op], [acc], summarize=self.BATCH_SIZE, message="Acuraccy: ")
            return op


model_trainer = ModelTrainer()
prediction_client = PredictionClient()

# get model parameters as private tensors from model owner
params = tfe.define_private_input('model-trainer', model_trainer.provide_input, masked=True)  # pylint: disable=E0632

# we'll use the same parameters for each prediction so we cache them to avoid re-training each time
params = tfe.cache(params)

# get prediction input from client
x, y = tfe.define_private_input('prediction-client', prediction_client.provide_input, masked=True)  # pylint: disable=E0632

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
    sess.run(tfe.global_caches_updater(), tag='training')

    for _ in range(5):
        print("Predicting")
        sess.run(prediction_op, tag='prediction')
