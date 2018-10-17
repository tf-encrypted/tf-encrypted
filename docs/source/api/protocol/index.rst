`protocol`
===========

In `tf-encrypted`, a protocol represents a certain type of cryptographic protocol
to achieve security.

The goal is to allow you to easily play with or use different cryptographic methods
by simply changing the protocol.

.. code-block:: python

  import tf_encrypted as tfe

  tfe.set_protocol(tfe.protocol.SecureNN())
  tfe.set_protocol(tfe.protocol.Pond())


.. automodule:: tf_encrypted.protocol.protocol
  :members:

.. toctree::
   :maxdepth: 5
   :caption: Classes

   protocol
   pond
   securenn
