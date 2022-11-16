import time
from abc import abstractmethod
from typing import Iterator

import numpy as np
import tensorflow as tf
from tensorflow.keras import utils

import tf_encrypted as tfe
from tf_encrypted.keras.engine.base_layer import Layer
from tf_encrypted.protocol.protocol import TFEPrivateTensor


class BaseModel(Layer):
    """
    Base Model class.
    This is the class from which all Models inherit.
    A model is a special layer with training and inference features.
    Users will just instantiate a model and then treat it as a callable.
    We recommend that descendants of `BaseModel` implement the following methods:
    * `__init__()`: Save configuration in member variables.
    * `call()`: model forward propagate.
    * `backward()`: model backward propagate.
    * `compile()`: set optimizer and loss.
    """

    def __init__(self, name=None):
        super(BaseModel, self).__init__(name)

        self._loss = None
        self._optimizer = None
        self.train_function = None
        self.predict_function = None
        self.test_function = None

    def __call__(self, inputs, *args, **kargs):
        with tf.name_scope(self._name):
            outputs = self.call(inputs, *args, **kargs)

        return outputs

    @abstractmethod
    def call(self, inputs, training=None, mask=None):
        """Implement model's forward propagation"""

    @abstractmethod
    def backward(self, d_y):
        """Implement model's backward propagation"""

    @abstractmethod
    def compile(self, optimizer, loss):
        """Configures the model's optimizer and loss"""

    def fit_batch(self, x, y):
        """Trains the model on a single batch.

        Arguments:
          x: Private tensor of training data
          y: Private tensor of target (label) data
        """

        y_pred = self.call(x)
        dy = self._loss.grad(y, y_pred)
        self.backward(dy)
        loss = self._loss(y, y_pred)

        return loss.reveal().to_native()

    def make_train_function(self):
        @tfe.function
        def train_step(input_x, input_y):
            return self.fit_batch(input_x, input_y)

        return train_step

    def fit(
        self, x=None, y=None, batch_size=32, epochs=1, verbose=1, steps_per_epoch=None
    ):
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        Args:
            x: Input data. It could be:
              - A private tf-encrypted tensor.
              - A generator returning `(inputs, targets)`.
            y: Target data. Like the input data `x`,
              it could be private tf-encrypted tensor.
              If `x` is a generator, `y` should not be
              specified (since targets will be obtained from `x`).
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of generators, since it generate batches.
            epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y` data provided
                (unless the `steps_per_epoch` flag is set to something other than None).
            verbose: 'auto', 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
                'auto' defaults to 1 for most cases, but 2 when used with
                `ParameterServerStrategy`. Note that the progress bar is not
                particularly useful when logged to a file, so verbose=2 is
                recommended when not running interactively (eg, in a production
                environment).
            steps_per_epoch: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring one epoch finished and starting the next epoch.
                When training with input tensors, the default `None` is equal to
                the number of samples in your dataset divided by
                the batch size, or 1 if that cannot be determined.
        """

        data_iter = data_wrap(x, y, batch_size)
        if self.train_function is None:
            self.train_function = self.make_train_function()
        for e in range(epochs):
            print("Epoch {}/{}".format(e + 1, epochs))
            progbar = utils.Progbar(steps_per_epoch, verbose=verbose)
            for index, (input_x, input_y) in enumerate(data_iter):
                start = time.time()
                current_loss = self.train_function(input_x, input_y)
                end = time.time()
                progbar.add(1, values=[("loss", current_loss), ("time", end - start)])
                if steps_per_epoch is not None and index + 1 >= steps_per_epoch:
                    break

    def make_predict_function(self, reveal=True):
        @tfe.function
        def predict_step(input_x):
            y_pred = self.call(input_x)
            if reveal:
                if isinstance(y_pred, list):
                    y_pred = [y.reveal().to_native() for y in y_pred]
                else:
                    y_pred = y_pred.reveal().to_native()
            return y_pred

        return predict_step

    def predict(self, x, batch_size=32, reveal=True):
        y_preds = []
        data_iter = data_wrap(x, None, batch_size)
        if self.predict_function is None:
            self.predict_function = self.make_predict_function(reveal)

        for input_x in data_iter:
            y_pred = self.predict_function(input_x)
            y_preds.append(y_pred)

        if reveal:
            concat = np.concatenate
        else:
            concat = tfe.concat

        if isinstance(y_preds[0], list):
            y_pred = [[] for i in range(len(y_preds[0]))]
            for i in range(len(y_preds)):
                for j in range(len(y_preds[0])):
                    y_pred[j].append(y_preds[i][j])
            for i in range(len(y_preds[0])):
                y_pred[i] = concat(y_pred[i], axis=0)
        else:
            y_pred = concat(y_preds, axis=0)
        return y_pred

    def make_test_function(self):
        @tfe.function
        def test_step(x):
            return self.call(x).reveal().to_native()

        return test_step

    def evaluate(self, x=None, y=None, batch_size=None, steps=None, metrics=None):

        if self.test_function is None:
            self.test_function = self.make_test_function()

        if metrics is None:
            return {}
        result = {}
        for metric in metrics:
            if metric == "categorical_accuracy":
                result[metric] = lambda y_true, y_pred: tf.reduce_mean(
                    tf.keras.metrics.categorical_accuracy(y_true, y_pred)
                )
            if metric == "binary_accuracy":
                result[metric] = lambda y_true, y_pred: tf.reduce_mean(
                    tf.keras.metrics.binary_accuracy(y_true, y_pred)
                )
        y_preds = []
        y_trues = []
        data_iter = data_wrap(x, y, batch_size)
        for index, (input_x, input_y) in enumerate(data_iter):
            y_pred = self.test_function(input_x)
            y_preds.append(y_pred)
            y_trues.append(input_y.reveal().to_native())
            if steps is not None and index + 1 >= steps:
                break

        y_pred = np.concatenate(y_preds)
        y_true = np.concatenate(y_trues)
        for metric in result.keys():
            result[metric] = result[metric](y_true, y_pred)

        return result


def data_wrap(x, y=None, batch_size=32):
    if isinstance(x, Iterator):
        return x
    elif isinstance(x, TFEPrivateTensor) and isinstance(y, TFEPrivateTensor):

        def iter_over_data(x_data, y_data, batch_size):
            start_index = 0
            end_index = batch_size
            while start_index < x_data[0].shape[0]:
                yield (x_data[start_index:end_index], y_data[start_index:end_index])
                start_index += batch_size
                end_index += batch_size

        data_iter = iter_over_data(x, y, batch_size)
        return data_iter
    elif isinstance(x, list) and isinstance(y, TFEPrivateTensor):

        def iter_over_data(x_data, y_data, batch_size):
            start_index = 0
            end_index = batch_size
            while start_index < x_data[0].shape[0]:
                x_batch_data = []
                for x in x_data:
                    x_batch_data.append(x[start_index:end_index])
                yield (x_batch_data, y_data[start_index:end_index])
                start_index += batch_size
                end_index += batch_size

        data_iter = iter_over_data(x, y, batch_size)
        return data_iter
    elif isinstance(x, TFEPrivateTensor):

        def iter_over_data(x_data, batch_size):
            start_index = 0
            end_index = batch_size
            while start_index < x_data.shape[0]:
                yield x_data[start_index:end_index]
                start_index += batch_size
                end_index += batch_size

        data_iter = iter_over_data(x, batch_size)
        return data_iter
    elif isinstance(x, list):

        def iter_over_data(x_data, batch_size):
            start_index = 0
            end_index = batch_size
            while start_index < x_data[0].shape[0]:
                x_batch_data = []
                for x in x_data:
                    x_batch_data.append(x[start_index:end_index])
                yield x_batch_data
                start_index += batch_size
                end_index += batch_size

        data_iter = iter_over_data(x, batch_size)
        return data_iter
    else:
        raise ValueError(
            "Inputs could be two private tfe tensor \
            for 'x' and 'y' or generater generate ('x', 'y')."
        )
