from __future__ import absolute_import
import sys
from typing import List, Tuple
import math

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


def weight_variable(shape, gain):
    """weight_variable generates a weight variable of a given shape."""
    if len(shape) == 2:
        fan_in, fan_out = shape
    elif len(shape) == 4:
        h, w, c_in, c_out = shape
        fan_in = h * w * c_in
        fan_out = h * w * c_out
    r = gain * math.sqrt(6 / (fan_in + fan_out))
    initial = tf.random_uniform(shape, minval=-r, maxval=r)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial)


conv2d = lambda x, w, s: tf.nn.conv2d(x, w, strides=[1, s, s, 1], padding='VALID')
pooling = lambda x: tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')


class ModelTrainer():

    BATCH_SIZE = 256
    ITERATIONS = 60000 // BATCH_SIZE
    EPOCHS = 2
    LEARNING_RATE = 3e-3
    IN_DIM = 28
    KERNEL = 5
    STRIDE = 1
    IN_CHANNELS = 1
    HIDDEN_CHANNELS = 16
    HIDDEN_FC1 = 256
    HIDDEN_FC2 = 100
    OUT_N = 10

    def cond(self, i: tf.Tensor, max_iter: tf.Tensor, nb_epochs: tf.Tensor, avg_loss: tf.Tensor) -> tf.Tensor:
        is_end_epoch = tf.equal(i % max_iter, 0)
        to_continue = tf.cast(i < max_iter * nb_epochs, tf.bool)

        def true_fn() -> tf.Tensor:
            to_continue = tf.print("avg_loss: ", avg_loss)
            return to_continue

        def false_fn() -> tf.Tensor:
            return to_continue

        return tf.cond(is_end_epoch, true_fn, false_fn)

    def build_training_graph(self, training_data) -> List[tf.Tensor]:

        # model parameters and initial values
        Wconv1 = weight_variable([self.KERNEL,
                                  self.KERNEL,
                                  self.IN_CHANNELS,
                                  self.HIDDEN_CHANNELS], 1.)
        bconv1 = bias_variable([1, 1, self.HIDDEN_CHANNELS])
        Wconv2 = weight_variable([self.KERNEL,
                                  self.KERNEL,
                                  self.HIDDEN_CHANNELS,
                                  self.HIDDEN_CHANNELS], 1.)
        bconv2 = bias_variable([1, 1, self.HIDDEN_CHANNELS])
        Wfc1 = weight_variable([self.HIDDEN_FC1, self.HIDDEN_FC2], 1.)
        bfc1 = bias_variable([self.HIDDEN_FC2])
        Wfc2 = weight_variable([self.HIDDEN_FC2, self.OUT_N], 1.)
        bfc2 = bias_variable([self.OUT_N])
        params = [Wconv1, bconv1, Wconv2, bconv2, Wfc1, bfc1, Wfc2, bfc2]

        # optimizer and data pipeline
        optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE)

        # training loop
        def loop_body(i: tf.Tensor, max_iter: tf.Tensor, nb_epochs: tf.Tensor, avg_loss: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

            # get next batch
            x, y = training_data.get_next()

            # model construction
            x = tf.reshape(x, [-1, self.IN_DIM, self.IN_DIM, 1])
            layer1 = pooling(tf.nn.relu(conv2d(x, Wconv1, self.STRIDE) + bconv1))
            layer2 = pooling(tf.nn.relu(conv2d(layer1, Wconv2, self.STRIDE) + bconv2))
            layer2 = tf.reshape(layer2, [-1, self.HIDDEN_FC1])
            layer3 = tf.nn.relu(tf.matmul(layer2, Wfc1) + bfc1)
            logits = tf.matmul(layer3, Wfc2) + bfc2

            loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=y))

            is_end_epoch = tf.equal(i % max_iter, 0)

            def true_fn() -> tf.Tensor:
                return loss

            def false_fn() -> tf.Tensor:
                return (tf.cast(i - 1, tf.float32) * avg_loss + loss) / tf.cast(i, tf.float32)

            with tf.control_dependencies([optimizer.minimize(loss)]):
                return i + 1, max_iter, nb_epochs, tf.cond(is_end_epoch, true_fn, false_fn)

        loop, _, _, _ = tf.while_loop(self.cond, loop_body, [0, self.ITERATIONS, self.EPOCHS, 0.])

        # return model parameters after training
        loop = tf.print("Training complete", loop)
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
    HIDDEN_FC1 = 256

    def provide_input(self) -> List[tf.Tensor]:
        with tf.name_scope('loading'):
            prediction_input, expected_result = get_data_from_tfrecord("./data/test.tfrecord", self.BATCH_SIZE).get_next()
 
        with tf.name_scope('pre-processing'):
            prediction_input = tf.reshape(prediction_input, shape=(self.BATCH_SIZE, 784))
            expected_result = tf.reshape(expected_result, shape=(self.BATCH_SIZE,))

        return [prediction_input, expected_result]

    def receive_output(self, likelihoods: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
        with tf.name_scope('post-processing'):
            prediction = tf.argmax(likelihoods, axis=1)
            eq_values = tf.equal(prediction, tf.cast(y_true, tf.int64))
            acc = tf.reduce_mean(tf.cast(eq_values, tf.float32))

            op = tf.print('Expected:', y_true, '\nActual:', prediction, '\nAccuracy:', acc)
            #op = tf.print('Expected:', likelihoods)

            return op


model_trainer = ModelTrainer()
prediction_client = PredictionClient()

# get model parameters as private tensors from model owner
params = tfe.define_private_input('model-trainer', model_trainer.provide_input, masked=True)  # pylint: disable=E0632

# we'll use the same parameters for each prediction so we cache them to avoid re-training each time
params = tfe.cache(params)

# get prediction input from client
x, y = tfe.define_private_input('prediction-client', prediction_client.provide_input, masked=True)  # pylint: disable=E0632

# helpers
# conv = lambda x, w: tfe.conv2d(x, w, strides=1, padding='VALID')
# pool = lambda x: tfe.avgpool2d(x, pool_size=(2, 2), strides=(2, 2), padding='VALID')


# compute prediction
Wconv1, bconv1, Wconv2, bconv2, Wfc1, bfc1, Wfc2, bfc2 = params

cv1_tfe = tfe.layers.Conv2D(
        input_shape=[-1, 28, 28, 1], filter_shape=[5, 5, 1, 16],
        strides=1,
        padding='VALID',
        channels_first=False
)

cv1_tfe.initialize(initial_weights=Wconv1)

cv2_tfe = tfe.layers.Conv2D(
        input_shape=[-1, 12, 12, 16], filter_shape=[5, 5, 16, 16],
        strides=1,
        padding='VALID',
        channels_first=False
)

cv2_tfe.initialize(initial_weights=Wconv2)

avg1 = tfe.layers.AveragePooling2D(input_shape=[-1, 24, 24, 16],
                                    pool_size=(2, 2),
                                    strides=(2,2),
                                    padding='VALID',
                                    channels_first=False)

avg2 = tfe.layers.AveragePooling2D(input_shape=[-1, 8, 8, 16],
                                    pool_size=(2, 2),
                                    strides=(2,2),
                                    padding='VALID',
                                    channels_first=False)

x = tfe.reshape(x, [-1, 28, 28, 1])

bconv1 = tfe.reshape(bconv1, [1, 1, -1])
bconv2 = tfe.reshape(bconv2, [1, 1, -1])
layer1 = avg1.forward(tfe.relu(cv1_tfe.forward(x) + bconv1))
layer2 = avg2.forward(tfe.relu(cv2_tfe.forward(layer1) + bconv2))

layer2 = tfe.reshape(layer2, [-1, ModelTrainer.HIDDEN_FC1])
layer3 = tfe.relu(tfe.matmul(layer2, Wfc1) + bfc1)
logits = tfe.matmul(layer3, Wfc2) + bfc2

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
