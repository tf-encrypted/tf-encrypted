import sys
from typing import List

import tensorflow as tf
import tf_encrypted as tfe

from convert import decode

if len(sys.argv) > 1:
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

    BATCH_SIZE = 30
    ITERATIONS = 60000 // BATCH_SIZE
    EPOCHS = 1

    def __init__(self, player: tfe.player.Player) -> None:
        self.player = player

    def build_data_pipeline(self):

        def normalize(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            return image, label

        dataset = tf.data.TFRecordDataset(["./data/train.tfrecord"])
        dataset = dataset.map(decode)
        dataset = dataset.map(normalize)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.BATCH_SIZE)

        iterator = dataset.make_one_shot_iterator()
        return iterator

    def build_training_graph(self, training_data) -> List[tf.Tensor]:

        # model parameters and initial values
        w0 = tf.Variable(tf.random_normal([28 * 28, 512]))
        b0 = tf.Variable(tf.zeros([512]))
        w1 = tf.Variable(tf.random_normal([512, 10]))
        b1 = tf.Variable(tf.zeros([10]))

        # optimizer and data pipeline
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

        # training loop
        def loop_body(i):

            # get next batch
            x, y = training_data.get_next()

            # model construction
            layer0 = tf.matmul(x, w0) + b0
            layer1 = tf.nn.sigmoid(layer0)
            layer2 = tf.matmul(layer1, w1) + b1

            predictions = layer2
            loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=predictions, labels=y))
            with tf.control_dependencies([optimizer.minimize(loss)]):
                return i + 1

        loop = tf.while_loop(lambda i: i < self.ITERATIONS * self.EPOCHS, loop_body, (0,))

        # return model parameters after training
        loop = tf.Print(loop, [], message="Training complete")
        with tf.control_dependencies([loop]):
            return [w0.read_value(), b0.read_value(), w1.read_value(), b1.read_value()]

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
            image = tf.cast(image, tf.float32) / 255.0
            return image, label

        dataset = tf.data.TFRecordDataset(["./data/test.tfrecord"])
        dataset = dataset.map(decode)
        dataset = dataset.map(normalize)
        dataset = dataset.batch(self.BATCH_SIZE)

        iterator = dataset.make_one_shot_iterator()
        return iterator

    def provide_input(self) -> tf.Tensor:
        with tf.name_scope('loading'):
            prediction_input, expected_result = self.build_data_pipeline().get_next()
            prediction_input = tf.Print(prediction_input, [expected_result], summarize=self.BATCH_SIZE, message="EXPECT ")

        with tf.name_scope('pre-processing'):
            prediction_input = tf.reshape(prediction_input, shape=(self.BATCH_SIZE, 28 * 28))

        return prediction_input

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
    w0, b0, w1, b1 = params
    layer0 = prot.matmul(x, w0) + b0
    layer1 = prot.sigmoid(layer0 * 0.1)  # input normalized to avoid large values
    logits = prot.matmul(layer1, w1) + b1

    # send prediction output back to client
    prediction_op = prot.define_output(prediction_client.player, [logits], prediction_client.receive_output)


target = sys.argv[2] if len(sys.argv) > 2 else None
with tfe.Session(target) as sess:

    print("Init")
    sess.run(tf.global_variables_initializer(), tag='init')

    print("Training")
    sess.run(tfe.global_caches_updator(), tag='training')

    for _ in range(5):
        print("Predicting")
        sess.run(prediction_op, tag='prediction')
