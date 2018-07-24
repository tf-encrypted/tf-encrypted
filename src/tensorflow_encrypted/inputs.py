
import numpy as np
import tensorflow as tf
import ops as tfe
from protocol import Player

class InputProvider(Player):

    def send_data(self, mask=True):

        with tf.device(self.device_name):

            with tf.name_scope('input'):

                with tf.name_scope('prepare'):

                    raw_x, raw_y = self._build_data_graph()

                    # this helps further on so enforce it for now
                    assert raw_x.shape.is_fully_defined()
                    assert raw_y.shape.is_fully_defined()

                    encoded_x = tfe.encode_and_decompose(raw_x)
                    encoded_y = tfe.encode_and_decompose(raw_y)

            if mask:
                # we are asked to pre-mask the data as well
                masked_x = tfe.local_mask(tfe.Tensor(encoded_x))
                masked_y = tfe.local_mask(tfe.Tensor(encoded_y))
                return masked_x, masked_y

            else:
                shared_x = tfe.PrivateTensor(*tfe.share(encoded_x))
                shared_y = tfe.PrivateTensor(*tfe.share(encoded_y))
                return shared_x, shared_y

    def _build_data_graph(self):
        """
        This is where loading and preprocessing of data happens,
        batching is taken care of later. Return type should be tf.Tensor.
        """
        raise NotImplementedError()

class NumpyInputProvider(InputProvider):

    def _build_data_graph(self):
        data_generator = self._build_data_generator()

        def wrapper():
            
            # call subclass' data generator
            X, Y = data_generator()

            # combine result into single tensor to get around `tf.py_func` limitation
            XY = np.hstack((X, Y))
            return XY

        xy = tf.py_func(wrapper, [], np.float32)
        x, y = tf.split(xy, self.num_cols, axis=1)
        x = tf.reshape(x, (self.num_rows, self.num_cols[0]))
        y = tf.reshape(y, (self.num_rows, self.num_cols[1]))
        return x, y

    def _build_data_generator(self):
        """
        This is where loading of data as a NumPy array is suppose to happen.
        Function is executed on `device_name`.
        """
        raise NotImplementedError()


class FakeInputProvider(NumpyInputProvider):

    def __init__(self, device_name):
        super(FakeInputProvider, self).__init__(device_name)

    @property
    def num_rows(self):
        return 1000

    @property
    def num_cols(self):
        return (2,1)

    def _build_data_generator(self):

        def generate_fake_training_data():
            np.random.seed(42)

            data_size = self.num_rows

            # generate features
            X0 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], data_size//2)
            X1 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], data_size//2)
            X = np.vstack((X0, X1)).astype(np.float32)

            # generate labels
            Y0 = np.zeros(data_size//2).reshape(-1, 1)
            Y1 = np.ones(data_size//2).reshape(-1, 1)
            Y = np.vstack((Y0, Y1)).astype(np.float32)

            # shuffle
            perm = np.random.permutation(len(X))
            X = X[perm]
            Y = Y[perm]

            return X, Y

        return generate_fake_training_data

