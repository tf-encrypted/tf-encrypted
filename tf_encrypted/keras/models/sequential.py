"""Sequential model API."""
import tensorflow as tf
from tensorflow.keras import utils

import tf_encrypted as tfe
from tf_encrypted.keras import backend as KE
from tf_encrypted.keras import optimizers
from tf_encrypted.keras.engine.base_layer import Layer
from tf_encrypted.keras.engine.input_layer import InputLayer, Input
from tf_encrypted.protocol.pond import PondPrivateTensor


class Sequential(Layer):
    """Model defined by a stack of layers in sequence."""

    def __init__(self, layers=None, name=None):
        super(Sequential, self).__init__(name=name)

        self._layers = []

        # Add to the model any layers passed to the constructor.
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        """Adds a layer instance on top of the layer stack.

    Arguments:
      layer: layer instance.

    Raises:
      TypeError: If `layer` is not a layer instance.
      ValueError: In case the `layer` argument does not
        know its input shape.
      ValueError: In case the `layer` argument has
        multiple output tensors, or is already connected
        somewhere else (forbidden in `Sequential` models).
    """
        if not isinstance(layer, Layer):
            raise TypeError(
                "The added layer must be "
                "an instance of class Layer. "
                "Found: " + str(layer)
            )
        self.built = False
        set_inputs = False
        if not self._layers:
            if isinstance(layer, InputLayer):
                raise ValueError(
                    "Do not manually define an InputLayer in your "
                    "tfe.keras.Sequential model."
                )

            batch_shape = layer._batch_input_shape  # pylint: disable=protected-access

            # Instantiate an input layer.
            x = Input(batch_shape=batch_shape, name=layer.name + "_input")
            # This will build the current layer
            # and create the node connecting the current layer
            # to the input layer we just created.
            y = layer(x)

            # If an input layer (placeholder) is available.
            if isinstance(y, (tuple, list)):
                raise ValueError(
                    "All layers in a Sequential model "
                    "should have a single output tensor. "
                    "For multi-output layers, "
                    "use the functional API."
                )
            self.outputs = [y]

        elif self.outputs:
            # If the model is being built continuously on top of an input layer:
            # refresh its output.
            output_tensor = layer(self.outputs[0])
            if isinstance(output_tensor, list):
                raise TypeError(
                    "All layers in a Sequential model "
                    "should have a single output tensor. "
                    "For multi-output layers, "
                    "use the functional API."
                )
            self.outputs = [output_tensor]
        if set_inputs:
            self.built = True
        else:
            self._layers.append(layer)

    def call(
        self, inputs, training=None, mask=None,
    ):  # pylint: disable=arguments-differ
        if training is not None:
            raise NotImplementedError()
        if mask is not None:
            raise NotImplementedError()
        outputs = inputs  # handle the corner case where self.layers is empty
        for layer in self.layers:
            # During each iteration, `inputs` are the inputs to `layer`, and `outputs`
            # are the outputs of `layer` applied to `inputs`. At the end of each
            # iteration `inputs` is set to `outputs` to prepare for the next layer.
            outputs = layer(inputs)

            # `outputs` will be the inputs to the next layer.
            inputs = outputs

        return outputs

    @property
    def layers(self):
        """Historically, `sequential.layers` only returns layers that were added
    via `add`, and omits the auto-generated `InputLayer` that comes at the
    bottom of the stack."""
        layers = self._layers
        if layers and isinstance(layers[0], InputLayer):
            return layers[1:]
        return layers[:]

    def backward(self, d_y):
        for layer in reversed(self.layers):
            grad_weights, d_y = layer.backward(d_y)
            self._optimizer.apply_gradients(layer.weights, grad_weights)

    def compile(self, optimizer, loss):
        """Configures the model for training.

      Arguments:
        optimizer: Optimizer instance
        loss: Objective function
    """
        self._optimizer = optimizers.get(optimizer)
        self._loss = loss
        assert self._optimizer is not None, "An optimizer must be specified."
        assert self._loss is not None, "A loss must be specified."

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

        sess = KE.get_session()
        self._current_loss = sess.run(loss.reveal())

    def fit(self, x, y, epochs=1, steps_per_epoch=1):
        """Trains the model for a given number of epochs
    (iterations on a dataset).

    Arguments:
      x: Private tensor of training data
      y: Private tensor of target (label) data
      epochs: Integer. Number of epochs to train the model.
      steps_per_epoch: Integer. Total number of steps (batches of samples)
        before declaring one epoch finished and starting the next epoch.
    """
        assert isinstance(x, PondPrivateTensor), type(x)
        assert isinstance(y, PondPrivateTensor), type(y)

        # Initialize variables before starting to train
        sess = KE.get_session()
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            print("Epoch {}/{}".format(e + 1, epochs))
            batch_size = x.shape.as_list()[0]
            progbar = utils.Progbar(batch_size * steps_per_epoch)
            for _ in range(steps_per_epoch):
                self.fit_batch(x, y)
                progbar.add(batch_size, values=[("loss", self._current_loss)])

    def set_weights(self, weights, sess=None):
        """Sets the weights of the model.

    Arguments:
      weights: A list of Numpy arrays with shapes and types
        matching the output of model.get_weights()
      sess: tfe.Session instance.
    """

        if not sess:
            sess = KE.get_session()

        # Updated weights for each layer
        for layer in self.layers:
            num_param = len(layer.weights)
            if num_param == 0:
                continue
            layer_weights = weights[:num_param]

            layer.set_weights(layer_weights, sess)

            weights = weights[num_param:]

    @classmethod
    def from_config(cls, config):
        """Instantiates a TFE Keras model from its config.

    Arguments:
      config: Configuration dictionary matching the output of
        model.get_weights().

    Returns:
        A TFE Keras Sequential instance.
    """
        tfe_model = model_from_config(config)

        return tfe_model


def model_from_config(config):
    """Instantiates a TFE Keras model from its config.

  Arguments:
    config: Configuration dictionary matching the output of
        model.get_weights().

  Returns:
    A TFE Keras Sequential instance.
  """

    tfe_model = tfe.keras.Sequential([])

    for k_l_c in config["layers"]:
        tfe_layer = _instantiate_tfe_layer(k_l_c)
        tfe_model.add(tfe_layer)

    return tfe_model


def clone_model(model):
    """Clone any tf.keras.Model into a tfe.keras.Sequenial model.

  Arguments:
    model: tf.keras.Sequential or tf.keras.Model instance.

  Returns:
    A TFE Keras model instance reproducing the behavior of the
    original model using newly instantiated weights.
  """

    config = model.get_config()
    weights = model.get_weights()

    tfe_model = model_from_config(config)
    tfe_model.set_weights(weights)

    return tfe_model


def _instantiate_tfe_layer(keras_layer_config):
    """instantiate TFE layer based on Keras layer config.

  Arguments:
    keras_layer_config: result of layer.get_config().

  Returns:
    A TFE Keras layer instance reproducing the behavior of the
    original Keras layer.
  """

    # Identify tf.keras layer type, and grab the corresponding tfe.keras layer
    keras_layer_type = keras_layer_config["class_name"]
    try:
        tfe_layer_cls = getattr(tfe.keras.layers, keras_layer_type)
    except AttributeError:
        # TODO: rethink how we warn the user about this, maybe codegen a list of
        #       supported layers in a doc somewhere
        raise RuntimeError(
            "TF Encrypted does not yet support the {lcls} layer.".format(
                lcls=keras_layer_type
            )
        )

    # get layer config to instiate the tfe layer with the right parameters
    config = keras_layer_config["config"]

    return tfe_layer_cls(**config)
