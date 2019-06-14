"""Sequential model API."""
import tensorflow as tf
from tf_encrypted.keras.engine.base_layer import Layer
from tf_encrypted.keras.engine.input_layer import InputLayer, Input
import tf_encrypted


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

      # Add layer weights to the  model weights list
      #self.weights += layer.weights

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

  def from_config(self, keras_config):

    self._rebuild_tfe_model(keras_config)

  def set_weights(self, weights_array, sess):
    print(self._layers)
    m_weights = []
    for l in self._layers:
      m_weights += l.weights

    for i, w in enumerate(m_weights):
      new_w = self.prot.define_private_variable(weights_array[i])
      sess.run(tf.global_variables_initializer())
      sess.run(self.prot.assign(w, new_w))

  def _rebuild_tfe_model(self, keras_config):
    """
    Rebuild the plaintext Keras model as a TF Encrypted Keras model
    using the keras configuration and the current TF Encrypted protocol
    and configuration.
    """
    for keras_layer in keras_config['layers']:
      tfe_layer = _instantiate_tfe_layer(keras_layer)
      self._layers.append(tfe_layer)

def _instantiate_tfe_layer(keras_layer):

# Identify tf.keras layer type, and grab the corresponding tfe.keras layer
  keras_layer_type = keras_layer['class_name']
  try:
    tfe_layer_cls = getattr(tf_encrypted.keras.layers, keras_layer_type)
  except AttributeError:
    # TODO: rethink how we warn the user about this, maybe codegen a list of
    #       supported layers in a doc somewhere
    raise RuntimeError(
      "TF Encrypted does not yet support the " "{lcls} layer.".format(lcls=keras_layer_type)
    )

  keras_layer_config = keras_layer['config']
  keras_layer_config.pop('dtype')

  return tfe_layer_cls(**keras_layer_config)
