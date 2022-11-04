"""BloomRegressor implementation and DataOwner helper."""
import numpy as np
import tensorflow as tf

import tf_encrypted as tfe


class BloomRegressor:
    """Secure multi-party linear regression at plaintext speed.

    Computes the necessary components of the normal equations solution to a
    linear regression."""

    def __init__(self):
        self.components = [
            "label_square",
            "covariate_label_product",
            "covariate_square",
        ]

    @classmethod
    def estimator_fn(cls, x_p, y_p):
        # Recall beta = np.inv(X.T @ X) * (X.T @ y)
        yy_p = tf.matmul(y_p, y_p, transpose_a=True)  # per-party y.T @ y
        xy_p = tf.matmul(x_p, y_p, transpose_a=True)  # per-party X.T @ y
        xx_p = tf.matmul(x_p, x_p, transpose_a=True)  # per-party X.T @ X
        return yy_p, xy_p, xx_p

    def fit(self, training_players, summary=0, validation_split=None):
        """Trains the linear regressor.

        Arguments:
          training_players: Data owners used for joint training. Must implement the
              compute_estimators as a tfe.local_computation.
          summary: Controls what kind of summary statistics are generated after the
              linear regression fit.
          validation_split: Mimics the behavior of the Keras validation_split kwarg.
        """
        if validation_split is not None:
            raise NotImplementedError()

        partial_estimators = [
            player.compute_estimators(self.estimator_fn) for player in training_players
        ]

        for attr, partial_estimator in zip(self.components, zip(*partial_estimators)):
            setattr(self, attr, tfe.add_n(partial_estimator))

        with tfe.Session() as sess:
            for k in self.components:
                op = getattr(self, k)
                setattr(self, k, sess.run(op.reveal()))

        tf_graph = tf.Graph()
        with tf_graph.as_default():
            self._inverted_covariate_square = tf.linalg.inv(self.covariate_square)
            self.coefficients = tf.matmul(
                self._inverted_covariate_square, self.covariate_label_product
            )

        with tf.Session(graph=tf_graph) as sess:
            for k in ["_inverted_covariate_square", "coefficients"]:
                setattr(self, k, sess.run(getattr(self, k)))

        if not summary:
            return self

        return self.summarize(summary_level=summary)

    def predict(self, x):
        raise NotImplementedError()

    def evaluate(self, testing_players):
        raise NotImplementedError()

    def __getattribute__(self, attr):
        # We will only use numpy arrays for storage, so we can safely lift them
        # into a tf.Tensor whenever they're requested.
        obj = super().__getattribute__(attr)
        if isinstance(obj, np.ndarray):
            return tf.constant(obj)
        return obj

    def summarize(self, summary_level):
        # TODO: coefficient variance
        # TODO: stderror
        # TODO: p-vals
        raise NotImplementedError()


class DataOwner:
    """Contains code meant to be executed by a data owner Player."""

    def __init__(
        self,
        player_name,
        num_features,
        training_set_size,
        test_set_size,
    ):
        self.player_name = player_name
        self.num_features = num_features
        self.training_set_size = training_set_size
        self.test_set_size = test_set_size

    def _build_training_data(self):
        """Preprocess training dataset

        Return single batch of training dataset
        """

        def cast(x, y):
            return tf.cast(x, tf.float32), y

        x_raw = tf.random.uniform(
            minval=-0.5, maxval=0.5, shape=[self.training_set_size, self.num_features]
        )
        target_coeffs = tf.random.normal(
            shape=[self.num_features, 1], mean=3, stddev=2.0
        )
        y_raw = tf.matmul(x_raw, target_coeffs)

        return cast(x_raw, y_raw)

    def _build_testing_data(self):
        """Preprocess testing dataset

        Return single batch of testing dataset
        """

        def cast(x, y):
            return tf.cast(x, tf.float32), tf.expand_dims(y, 0)

        x_raw = tf.random.uniform(
            minval=-0.5, maxval=0.5, shape=[self.training_set_size, self.num_features]
        )
        target_coeffs = tf.random.normal(shape=(self.num_features), mean=3, stddev=2.0)
        y_raw = tf.matmul(x_raw, target_coeffs)

        return cast(x_raw, y_raw)

    @tfe.local_computation
    def compute_estimators(self, estimator_fn):
        x, y = self._build_training_data()
        return estimator_fn(x, y)
