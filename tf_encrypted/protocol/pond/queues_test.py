# pylint: disable=missing-docstring
import unittest

import numpy as np
import tensorflow as tf

import tf_encrypted as tfe


class TestFIFO(unittest.TestCase):
    def test_fifo(self):
        prot = tfe.protocol.Pond()
        tfe.set_protocol(prot)

        shape = (10, 10)

        q = tfe.queue.FIFOQueue(
            capacity=10,
            shape=shape,
        )
        assert isinstance(q, tfe.protocol.pond.AdditiveFIFOQueue)

        raw = np.full(shape, 5)

        x = tfe.define_private_input("inputter", lambda: tf.convert_to_tensor(raw))
        assert isinstance(x, tfe.protocol.pond.PondPrivateTensor)

        q.enqueue(x)
        y = q.dequeue()
        assert isinstance(y, tfe.protocol.pond.PondPrivateTensor)
        assert y.backing_dtype == x.backing_dtype
        assert y.shape == x.shape
        res = y.reveal().to_native()
        np.testing.assert_array_equal(res, raw)


if __name__ == "__main__":
    unittest.main()
