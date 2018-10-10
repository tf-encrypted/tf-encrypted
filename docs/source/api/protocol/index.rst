`protocol`
========

In `tf-encrypted`, a protocol represents a certain type of cryptographic protocol
to achieve security.

The goal is that you can easily play with or use different cryptographic methods
by simply changing the protocol.

.. code-block:: python

  import tensorflow_encrypted as tfe

  tfe.set_protocol(tfe.protocol.SecureNN())


.. toctree::
   :maxdepth: 5
   :caption: Classes

   protocol
   pond
   securenn


.. automodule:: tensorflow_encrypted.protocol.protocol
  :members: set_protocol, get_protocol
