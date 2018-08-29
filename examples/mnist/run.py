from typing import List
import sys

import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe

from convert import decode


config = tfe.LocalConfig([
    'server0',
    'server1',
    'crypto_producer',
    'model_trainer',
    'prediction_client'
])

# config = tfe.RemoteConfig({
#     'server0': 'localhost:4440',
#     'server1': 'localhost:4441',
#     'crypto_producer': 'localhost:4442',
#     'model_trainer': 'localhost:4443',
#     'prediction_client': 'localhost:4444'
# })


if len(sys.argv) > 1:

    ####################################
    # assume we're running as a server #
    ####################################

    player_name = str(sys.argv[1])

    # pylint: disable=E1101
    server = config.server(player_name)
    server.start()
    server.join()

else:

    ##################################
    # assume we're running as master #
    ##################################

    class ModelTrainer(tfe.io.InputProvider):

        BATCH_SIZE = 30
        ITERATIONS = 60000//BATCH_SIZE
        EPOCHS = 1

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
            w0 = tf.Variable(tf.random_normal([28*28, 512]))
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


    class PredictionClient(tfe.io.InputProvider, tfe.io.OutputReceiver):

        BATCH_SIZE = 10

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

        def provide_input(self) -> List[tf.Tensor]:
            with tf.name_scope('loading'):
                prediction_input, expected_result = self.build_data_pipeline().get_next()
                prediction_input = tf.Print(prediction_input, [expected_result], summarize=10, message="EXPECT ")

            with tf.name_scope('pre-processing'):
                prediction_input = tf.reshape(prediction_input, shape=(self.BATCH_SIZE, 28*28))

            return [prediction_input]

        def receive_output(self, tensors: List[tf.Tensor]) -> tf.Operation:
            likelihoods, = tensors
            with tf.name_scope('post-processing'):
                prediction = tf.argmax(likelihoods, axis=1)
                op = tf.Print([], [prediction], summarize=self.BATCH_SIZE, message="ACTUAL ")
                return op


    model_trainer = ModelTrainer(config.get_player('model_trainer'))
    prediction_client = PredictionClient(config.get_player('prediction_client'))

    server0 = config.get_player('server0')
    server1 = config.get_player('server1')
    crypto_producer = config.get_player('crypto_producer')

    with tfe.protocol.Pond(server0, server1, crypto_producer) as prot:

        # get model parameters as private tensors from model owner
        w0, b0, w1, b1 = prot.define_private_input(model_trainer, masked=True) # pylint: disable=E0632

        # we'll use the same parameters for each prediction so we cache them to avoid re-training each time
        w0, b0, w1, b1 = prot.cache([w0, b0, w1, b1])

        # get prediction input from client
        x, = prot.define_private_input(prediction_client, masked=True) # pylint: disable=E0632

        # compute prediction
        layer0 = prot.dot(x, w0) + b0
        layer1 = prot.sigmoid(layer0 * 0.1) # input normalized to avoid large values
        layer2 = prot.dot(layer1, w1) + b1
        prediction = layer2

        # send prediction output back to client
        prediction_op = prot.define_output([prediction], prediction_client)

        with config.session() as sess:
            print("Init")
            tfe.run(sess, tf.global_variables_initializer(), tag='init')
            
            print("Training")
            tfe.run(sess, tfe.global_caches_updator(), tag='training')

            for _ in range(5):
                print("Predicting")
                tfe.run(sess, prediction_op, tag='prediction')
