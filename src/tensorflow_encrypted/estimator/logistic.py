from ..ops import *
# from ..training import *

class Classifier(object):
    pass

class LogisticClassifier(Classifier):

    def __init__(self, num_features, sess=None):
        self.num_features = num_features
        self.sess = sess or session()

    def set_parameters(self, weights, bias):
        pass

    def prepare_training_data(self, input_providers):
        # collect data from all input providers
        input_graphs = [ input_provider.send_data(mask=True) for input_provider in input_providers ]
        xs, ys = zip(*input_graphs)

        # combine
        combined_x = concat(xs)
        combined_y = concat(ys)

        # store in cache;
        # needed to avoid pulling again from input providers as these 
        # use random ops that force re-evaluation
        cache_initializers = []
        cache_updators = []
        cached_x = cache(combined_x, cache_initializers, cache_updators)
        cached_y = cache(combined_y, cache_initializers, cache_updators)

        # run
        self.sess.run(cache_initializers)
        self.sess.run(cache_updators)
        self.training_data = (cached_x, cached_y)

    def train(self, epochs=1, batch_size=10):
        assert self.training_data is not None, "No training data prepared"
        x, y = self.training_data

        # split into batches
        data_size = x.shape[0]
        num_batches = data_size // batch_size
        assert x.shape[0] == y.shape[0]
        assert data_size % batch_size == 0, "Batch size not matching size of training data {}".format(data_size)
        x_batched = split(x, num_batches)
        y_batched = split(y, num_batches)

        # execute training loop
        # TODO

    def predict(self, x):
        (input_x, y) = self._build_prediction_graph()

        x = x.reshape(1, self.num_features)
        y_pred = self.sess.run(
            reveal(y),
            feed_dict=encode_input((input_x, x))
        )
        return decode_output(y_pred)
        
    def _build_prediction_graphs(self):
        if self.prediction_graph is not None:
            return self.prediction_graph

        initial_weights = np.zeros((self.num_features, 1))
        initial_bias = np.zeros((1, 1))

        w = define_variable(initial_weights, name='w')
        b = define_variable(initial_bias, name='b')

        # build predict graph
        input_x, x = define_input((1, self.num_features), name='x')
        y = sigmoid(add(dot(x, w), b))
        self.prediction_graph = (input_x, y)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(cache_updators)

        # TODO
    