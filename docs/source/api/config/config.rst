`Config`
============

| Config determines how a session should run in tf-encrypted.
| There are two primary ways in which config can be used:

.. py:class:: tf_encrypted.config.LocalConfig
  :noindex:

and

.. py:class:: tf_encrypted.config.RemoteConfig
  :noindex:

As the name implies, `LocalConfig` is used to create a
local session. This is useful for quick debugging and prototyping.

`RemoteConfig` is more robust and is used to specify how a graph will run
in a production environment.  What machines are on which host, etc.

See class definitions for usage examples and more.

.. toctree::
   :caption: Classes

   local
   remote
