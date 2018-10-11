`SecureNN`
============

.. TODO: Link to real Maxpooling layers once they exist

SecureNN is an implementation from the `SecureNN paper`_.
SecureNN is an extension of the `Pond` protocol.  ie `SecureNN` is a superset of
the `SPDZ` protocol. The main difference between `SecureNN` and `SPDZ` is exact
:class:`~tensorflow_encrypted.layers.activation.Relu` and `Maxpooling` layers.
In SPDZ, `Maxpooling` is simply not supported, and :class:`~tensorflow_encrypted.layers.activation.Relu`
will be approximated.

.. TODO: @Yann could you write a blurb about the confidence interval we have?

Approximation can be quicker in some cases but it will break down when inputs
are sufficiently large. This requires users to implement workaround techniques such
as adding a :class:`~tensorflow_encrypted.layers.batchnorm.Batchnorm` layer before
a :class:`~tensorflow_encrypted.layers.activation.Relu`.

.. _SecureNN paper: https://eprint.iacr.org/2018/442.pdf

.. automodule:: tensorflow_encrypted.protocol.securenn
  :members:
