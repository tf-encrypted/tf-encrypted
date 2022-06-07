"""TFE Keras loss function"""
import tf_encrypted as tfe


class Loss:
    """Loss base class."""

    def __init__(self, loss_fn, **kwargs):

        self.loss_fn = loss_fn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        """Invokes the `LossFunctionWrapper` instance.
        Args:
            y_true: Ground truth values.
            y_pred: The predicted values.
        Returns:
            Loss values per sample.
        """
        return self.loss_fn(y_true, y_pred, **self._fn_kwargs)

    def __call__(self, y_true, y_pred):
        """Invokes the `Loss` instance.

    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.
    """
        return self.call(y_true, y_pred)


class BinaryCrossentropy(Loss):
    """Computes the cross-entropy loss between true labels and predicted labels.

    Args:
        from_logits: Whether to interpret `y_pred` as a tensor of
        [logit](https://en.wikipedia.org/wiki/Logit) values. By default we assume
            that `y_pred` contains probabilities (i.e., values in [0, 1]).
    """

    def __init__(self, from_logits=False, lazy_normalization=False):
        self.from_logits = from_logits
        self.lazy_normalization = lazy_normalization
        if from_logits:
            super(BinaryCrossentropy, self).__init__(binary_crossentropy_from_logits)
        else:
            super(BinaryCrossentropy, self).__init__(binary_crossentropy)

    def grad(self, y_true, y_pred):
        batch_size = y_true.shape.as_list()[0]
        batch_size_inv = 1 / batch_size
        if self.from_logits:
            y_pred = tfe.sigmoid(y_pred)
        grad = y_pred - y_true
        if not self.lazy_normalization:
            grad = grad * batch_size_inv
        return grad


def binary_crossentropy(y_true, y_pred):
    batch_size = y_true.shape.as_list()[0]
    batch_size_inv = 1 / batch_size
    out = y_true * tfe.log(y_pred)
    out += (1 - y_true) * tfe.log(1 - y_pred)
    out = out.negative()
    bce = out.reduce_sum(axis=0) * batch_size_inv
    return bce


def binary_crossentropy_from_logits(y_true, y_pred):
    y_pred = tfe.sigmoid(y_pred)
    return binary_crossentropy(y_true, y_pred)


class MeanSquaredError(Loss):
    """Computes the MSE loss between true labels and predicted labels."""

    def __init__(self):
        super(MeanSquaredError, self).__init__(mean_squared_error)

    def grad(self, y_true, y_pred):
        batch_size = y_true.shape.as_list()[0]
        batch_size_inv = 1 / batch_size
        return 2 * (y_pred - y_true) * batch_size_inv


def mean_squared_error(y_true, y_pred):
    batch_size = y_true.shape.as_list()[0]
    batch_size_inv = 1 / batch_size
    out = y_true - y_pred
    out = out.square()
    mse_loss = out.reduce_sum(axis=0) * batch_size_inv
    return mse_loss


class CategoricalCrossentropy(Loss):
    """
    See `tf.keras.losses.CategoricalCrossentropy`.
    """

    def __init__(self, from_logits=False, lazy_normalization=False):
        self.from_logits = from_logits
        self.lazy_normalization = lazy_normalization
        if from_logits:
            super(CategoricalCrossentropy, self).__init__(
                categorical_crossentropy_from_logits
            )
        else:
            super(CategoricalCrossentropy, self).__init__(categorical_crossentropy)

    def grad(self, y_true, y_pred):
        """
        Softmax grad reference:
        https://slowbreathing.github.io/articles/2019-05/softmax-and-cross-entropy
        """
        batch_size = y_true.shape.as_list()[0]
        batch_size_inv = 1 / batch_size
        if self.from_logits:
            y_pred = tfe.keras.activations.softmax(y_pred)
        grad = y_pred - y_true
        if not self.lazy_normalization:
            grad = grad * batch_size_inv
        return grad


def categorical_crossentropy(y_true, y_pred):
    batch_size = y_true.shape.as_list()[0]
    batch_size_inv = 1 / batch_size
    out = -y_true * tfe.log(y_pred)
    cce = out.reduce_sum() * batch_size_inv
    return cce


def categorical_crossentropy_from_logits(y_true, y_pred):
    y_pred = tfe.keras.activations.softmax(y_pred)
    return categorical_crossentropy(y_true, y_pred)
