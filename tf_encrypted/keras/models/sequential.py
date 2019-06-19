"""Sequential model API."""
from keras import backend as K

import tf_encrypted as tfe
from tf_encrypted.keras.engine.base_layer import Layer
from tf_encrypted.keras.engine.input_layer import InputLayer, Input


class Sequential(Layer):
  """Model defined by a stack of layers in sequence.

  TODO
  """
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
      raise TypeError('The added layer must be '
                      'an instance of class Layer. '
                      'Found: ' + str(layer))
    self.built = False
    set_inputs = False
    if not self._layers:
      if isinstance(layer, InputLayer):
        raise ValueError("Do not manually define an InputLayer in your "
                         "tfe.keras.Sequential model.")

      batch_shape = layer._batch_input_shape  # pylint: disable=protected-access

      # Instantiate an input layer.
      x = Input(
          batch_shape=batch_shape,
          name=layer.name + '_input')
      # This will build the current layer
      # and create the node connecting the current layer
      # to the input layer we just created.
      y = layer(x)

      # If an input layer (placeholder) is available.
      if isinstance(y, (tuple, list)):
        raise ValueError('All layers in a Sequential model '
                         'should have a single output tensor. '
                         'For multi-output layers, '
                         'use the functional API.')
      self.outputs = [y]

    elif self.outputs:
      # If the model is being built continuously on top of an input layer:
      # refresh its output.
      output_tensor = layer(self.outputs[0])
      if isinstance(output_tensor, list):
        raise TypeError('All layers in a Sequential model '
                        'should have a single output tensor. '
                        'For multi-output layers, '
                        'use the functional API.')
      self.outputs = [output_tensor]
    if set_inputs:
      self.built = True
    else:
      self._layers.append(layer)

  def call(self, inputs, training=None, mask=None):  # pylint: disable=arguments-differ
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

  def set_weights(self, weights, sess=None):
    """ Sets the weights of the model.
    Arguments:
      weights: A list of Numpy arrays with shapes and types
          matching the output of model.get_weights()
      sess: tfe session"""

    # Updated weights for each layer
    for layer in self.layers:
      num_param = len(layer.weights)
      layer_weights = weights[:num_param]
      # Define keras weights as private placeholder
      tfe_weights_pl = [tfe.define_private_placeholder(w.shape)
                        for w in layer_weights]
      # Assign new keras weights to existing weights defined by
      # default when tfe layer was instantiated
      if not sess:
        sess = K.get_session()
      for i, w in enumerate(layer.weights):
        fd = tfe_weights_pl[i].feed(layer_weights[i])
        sess.run(tfe.assign(w, tfe_weights_pl[i]), feed_dict=fd)

      weights = weights[num_param:]

  def from_config(self, keras_config):

    tfe_model = _rebuild_tfe_model(keras_config)

    return tfe_model

def model_from_config(keras_config):

  return _rebuild_tfe_model(keras_config)

def clone_model(model):
  """Clone any Sequential instance into TFE model"""

  config = model.get_config()
  weights = model.get_weights()

  tfe_model = model_from_config(config)

  sess = tfe.Session()
  tfe_model.set_weights(weights, sess)

  tfe_model._tfe_session = sess # pylint: disable=protected-access
  K.set_session(sess)

  return tfe_model

def _rebuild_tfe_model(keras_config):
  """
  Rebuild the plaintext Keras model as a TF Encrypted Keras model
  using the keras configuration and the current TF Encrypted protocol
  and configuration."""

  tfe_model = tfe.keras.Sequential([])

  for k_l_c in keras_config['layers']:
    tfe_layer = _instantiate_tfe_layer(k_l_c)
    tfe_model.add(tfe_layer)

  return tfe_model

def _instantiate_tfe_layer(keras_layer_config):
  """instantiate tfe layer based on layer keras config"""

  # Identify tf.keras layer type, and grab the corresponding tfe.keras layer
  keras_layer_type = keras_layer_config['class_name']
  try:
    tfe_layer_cls = getattr(tfe.keras.layers, keras_layer_type)
  except AttributeError:
    # TODO: rethink how we warn the user about this, maybe codegen a list of
    #       supported layers in a doc somewhere
    raise RuntimeError(
        "TF Encrypted does not yet support the " "{lcls} "
        "layer.".format(lcls=keras_layer_type)
    )

  # get layer config to instiate the tfe layer with the right parameters
  config = keras_layer_config['config']

  return tfe_layer_cls(**config)
