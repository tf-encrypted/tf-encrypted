"""Sequential model API."""
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
    print("seq__init__", tfe.protocol.protocol.__PROTOCOL__)

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
    print("seq_add", tfe.protocol.protocol.__PROTOCOL__)
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
