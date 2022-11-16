# pylint: disable=missing-docstring
import unittest

import numpy as np

from tf_encrypted.keras.engine.input_layer import Input

np.random.seed(42)


class TestInput(unittest.TestCase):
    def test_input(self):
        x = Input(shape=(2,), batch_size=1)
        result = x.reveal().to_native()
        np.testing.assert_allclose(result, np.zeros((1, 2)), rtol=0.0, atol=0.01)


if __name__ == "__main__":
    unittest.main()
