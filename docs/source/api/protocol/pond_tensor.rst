`PondTensor`
=============

`PondTensor` is the tensor type you will interact with most.

Generally, you will never instantiate a tensor directly, rather you create them
through `protocol`.

.. code-block :: python

    pondPrivateTensor = prot.define_private_variable(np.array([1,2,3,4]))
    pondPublicTensor = prot.define_private_variable(np.array([1,2,3,4]))
    

.. autoclass:: tensorflow_encrypted.protocol.pond.PondTensor
  :members:
