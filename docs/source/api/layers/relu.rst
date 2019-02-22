`Relu`
=============

Classicly, `Relu` computes the following on input:

.. code-block:: python

    Relu(x) = max(0, x)

In TF Encrypted, how `Relu` behaves will depend on the underlying protocol
you are using.

| With :class:`~tf_encrypted.protocol.pond.Pond`, `Relu` will be approximated using `Chebyshev Polynomial Approximation`_
| With :class:`~tf_encrypted.protocol.securenn.SecureNN`, `Relu` will behave as you expect (`Relu(x) = max(0, x)`)

.. _Chebyshev Polynomial Approximation: https://en.wikipedia.org/wiki/Chebyshev_polynomials


.. autoclass:: tf_encrypted.layers.activation.Relu
  :members:
