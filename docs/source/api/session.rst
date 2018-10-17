`Session`
==========

Session is an extension of `tf.Session` that lets the graph run in a secure manner.

The aim of `tf-encrypted` is to look as close to `TensorFlow` as possible.  With this goal in mind,
you get and use a session the same way that you're used to with Tensorflow::

    import tf_encrypted as tfe

    with tfe.Session() as sess:
      # sess.run like normal


See also the official `TensorFlow docs on Session`_.


.. _TensorFlow docs on Session: https://www.tensorflow.org/api_docs/python/tf/Session


.. autoclass:: tf_encrypted.session.Session
  :members:
