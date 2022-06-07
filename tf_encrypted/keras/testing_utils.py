"""Testing utilities for tfe.keras."""
import numpy as np
import tensorflow as tf

import tf_encrypted as tfe
from tf_encrypted.keras.models import Sequential


def agreement_test(
    tfe_layer_cls,
    kwargs=None,
    input_shape=None,
    input_data=None,
    rtol=1e-1,
    atol=1e-8,
    **tfe_kwargs,
):
    """Check agreement between a tf.keras layer and a tfe.keras layer.
    Arguments:
      tfe_layer_cls: Layer class object (from tfe.keras).
      kwargs: Optional dictionary of keyword arguments for instantiating the
          layers.
      input_shape: Input shape tuple.
      input_data: Numpy array of input data.
      expected_output: Shape tuple for the expected shape of the output.
      tfe_kwargs: Additional kwargs to supply the tfe.keras Layer not included in
          the argspec of the original tf.keras Layer object.
    Raises:
        ValueError: if `input_data is None and input_shape is None`.
    """
    input_shape, input_data = _sanitize_testing_args(input_shape, input_data)

    tf_layer_cls = getattr(tf.keras.layers, tfe_layer_cls.__name__)
    tfe_kwargs = {**kwargs, **tfe_kwargs}

    with tf.name_scope("TFE"):
        tfe_layer = tfe_layer_cls(**tfe_kwargs)
        x = tfe.define_private_variable(input_data)
        y = tfe_layer(x)

        d_y = tfe.define_private_variable(np.ones(y.shape))
        try:
            _, tfe_dx = tfe_layer.backward(d_y)
        except NotImplementedError:
            tfe_dx = None

        with tfe.Session() as sess:
            sess.run(tf.global_variables_initializer())
            actual_y = sess.run(y.reveal())
            if tfe_dx is not None:
                actual_dx = sess.run(tfe_dx.reveal())

    tf.reset_default_graph()

    with tf.name_scope("TF"):
        with tf.Session() as sess:
            tf_layer = tf_layer_cls(**kwargs)
            x = tf.Variable(input_data, dtype=tf.float32)
            y = tf_layer(x)
            sess.run(tf.global_variables_initializer())
            expected_y = sess.run(y)

            if tfe_dx is not None:
                dx = tf.gradients([y], [x])[0]
                expected_dx = sess.run(dx)

    np.testing.assert_allclose(actual_y, expected_y, rtol=rtol, atol=atol)
    if tfe_dx is not None:
        np.testing.assert_allclose(actual_dx, expected_dx, rtol=rtol, atol=atol)


def layer_test(layer_cls, kwargs=None, batch_input_shape=None, input_data=None):
    """Test routine for a layer with a single input and single output.
    Arguments:
      layer_cls: Layer class object.
      kwargs: Optional dictionary of keyword arguments for instantiating the
        layer.
      input_shape: Input shape tuple.
      input_dtype: Data type of the input data.
      input_data: Numpy array of input data.
    Returns:
      The output data (Numpy array) returned by the layer, for additional
      checks to be done by the calling code.
    Raises:
      ValueError: if `input_data is None and input_shape is None`.
    """
    input_shape, input_data = _sanitize_testing_args(batch_input_shape, input_data)

    # instantiation
    kwargs = kwargs or {}

    with tf.name_scope("TFE"):
        layer = layer_cls(batch_input_shape=input_shape, **kwargs)
        model = Sequential()
        model.add(layer)

        x = tfe.define_private_variable(input_data)
        model(x)


def _sanitize_testing_args(input_shape, input_data):
    """Construct appropriate values for input_shape and input_data whenever one
    is missing."""
    if input_data is None:
        if input_shape is None:
            raise ValueError("input_shape is None")
        input_data_shape = list(input_shape)
        for i, e in enumerate(input_data_shape):
            if e is None:
                input_data_shape[i] = 2
        input_data = 10 * np.random.random(input_data_shape)
    elif input_shape is None:
        input_shape = input_data.shape
    return input_shape, input_data
