`Session`
==========

Session is an extension of `tf.Session` that lets the graph run in a secure manner.

You get and use a session in the same manner that you are used to with Tensorflow::

    import tensorflow_encrypted as tfe

    with tfe.Session() as sess:
      # sess.run like normal

TODO -- More coming soon.

.. autoclass:: tensorflow_encrypted.session.Session
  :members:
