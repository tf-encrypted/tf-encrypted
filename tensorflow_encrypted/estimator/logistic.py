import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe

from ..ops import concat, cache, decode_output
# from ..training import *


class Classifier(object):
    pass


def training_loop(training_step, iterations, initial_weights, initial_bias, x, y):
    initial_w0, initial_w1 = initial_weights
    initial_b0, initial_b1 = initial_bias

    # TODO use initial_bias!

    def loop_op(i, w0, w1):
        w = PrivateTensor(w0, w1)
        # x = x_batched[i]
        # y = y_batched[i]
        new_w = training_step(w, x, y)
        return new_w.share0, new_w.share1

    _, final_w0, final_w1 = tf.while_loop(
        cond=lambda i, w0, w1: tf.less(i, iterations),
        body=lambda i, w0, w1: (i+1,) + loop_op(i, w0, w1),
        loop_vars=(0, initial_w0, initial_w1),
        parallel_iterations=1
    )

    final_weights = (final_w0, final_w1)
    final_bias = (initial_b0, initial_b1)  # TODO
    return final_weights, final_bias


class LogisticClassifier(Classifier):

    def __init__(self, prot, session, num_features):
        self.prot = prot
        self.sess = session
        self.num_features = num_features
        self.parameters = None

    def initialize_parameters(self):
        initial_weights_value = np.zeros(shape=(self.num_features, 1))
        initial_bias = np.zeros((1, 1))

        w = self.prot.define_private_variable(initial_weights_value, name='w')
        b = self.prot.define_private_variable(initial_bias, name='b')

        tfe.run(self.sess, self.prot.initializer, tag='init')
        self.parameters = (w, b)

    def prepare_training_data(self, input_providers):
        # collect data from all input providers
        input_graphs = [input_provider.send_data(mask=True) for input_provider in input_providers]
        xs, ys = zip(*input_graphs)

        # combine
        combined_x = concat(xs)
        combined_y = concat(ys)

        # store in cache;
        # needed to avoid pulling again from input providers as these
        # may use random ops that force re-evaluation
        cache_initializers = []
        cache_updators = []
        cached_x = cache(combined_x, cache_initializers, cache_updators)
        cached_y = cache(combined_y, cache_initializers, cache_updators)

        # execute
        tfe.run(self.sess, cache_initializers)
        tfe.run(self.sess, cache_updators, 'prepare')
        self.training_data = (cached_x, cached_y)

    def train(self, epochs=1, batch_size=10):
        assert self.training_data is not None, "No training data prepared"
        x, y = self.training_data

        # make sure we have weights to work with
        if self.parameters is None:
            self.initialize_parameters()

        # TODO[Morten] batching
        # split into batches
        # data_size = x.shape[0]
        # num_batches = data_size // batch_size
        # assert x.shape[0] == y.shape[0]
        # assert data_size % batch_size == 0, "Batch size not matching size of training data {}".format(data_size)
        # x_batched = split(x, num_batches)
        # y_batched = split(y, num_batches)

        learning_rate = .01

        # build training graph
        def training_step(w, x, y):
            batch_size = int(x.shape[0])

            with tf.name_scope('forward'):
                y_pred = self.prot.sigmoid(self.prot.dot(x, w))

            with tf.name_scope('backward'):
                error = self.prot.sub(y_pred, y)
                gradients = self.prot.scale(
                    self.prot.dot(self.prot.transpose(x), error), 1./batch_size
                )
                new_w = self.prot.sub(w, self.prot.scale(gradients, learning_rate))
                # new_b = ...
                return new_w  # new_b

        old_weights, old_bias = self.parameters
        training = training_loop(
            training_step=training_step,
            iterations=epochs,
            initial_weights=old_weights.unwrapped,
            initial_bias=old_bias.unwrapped,
            x=x.unmasked,
            y=y.unmasked
        )

        # TODO[Morten] need to assign new_weights and new_bias here into of returning them, no?

        new_weights, new_bias = tfe.run(self.sess, training, 'train')
        self.parameters = (
            self.prot.define_private_variable(*new_weights),
            PrivateTensor(*new_bias)
        )

    def _build_training_graph(self):
        pass

    def predict(self, x):
        (input_x, y) = self._build_prediction_graph()

        x = x.reshape(1, self.num_features)
        y_pred = self.sess.run(
            self.prot.reveal(y),
            feed_dict=tfe.encode_input((input_x, x))
        )
        return decode_output(y_pred)

    def _build_prediction_graphs(self):
        if self.prediction_graph is not None:
            return self.prediction_graph

        p = self.prot

        input_x = self.prot.define_private_placeholder((1, self.num_features), name='x')
        w, b = self.parameters
        y = p.sigmoid(p.add(w.dot(input_x) + b))

        self.prediction_graph = (input_x, y)
