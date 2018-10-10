`Config`
============

| Config determines how a session should run in tf-encrypted.
| There are two primary ways in which config can be used:
.. TODO: How do we link these to the real thing?
.. py:class:: tensorflow_encrypted.config.LocalConfig
and

.. py:class:: tensorflow_encrypted.config.RemoteConfig
As the name implies, `LocalConfig` is used to create a
local session. This is useful for quick debugging and prototyping.

`RemoteConfig` is more robust and is used to specify how a graph will run
in a production environment.  What machines are on which host, etc.

See class definitions for usage examples and more.

.. toctree::

   local
   remote
