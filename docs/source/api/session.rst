`Session`
==========

Session is an extension of `tf.Session` that lets the graph run in a secure manner.

You get and use a session the same way that you're used to with Tensorflow::

    import tf_encrypted as tfe

    with tfe.Session() as sess:
      # sess.run like normal

TODO -- More coming soon.

.. autoclass:: tf_encrypted.session.Session
  :members:
