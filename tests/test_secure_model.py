import unittest

import numpy as np
import tensorflow as tf
from tf_encrypted.private_model import secure_model


class TestSecureModel(unittest.TestCase):

    def test_secure_model(self):
        tf.random.set_random_seed(42)

        d = tf.keras.layers.Dense(1, input_shape=(10,), use_bias=False)
        model = tf.keras.Sequential([
            d
        ])

        input = np.ones((1, 10))
        output = model.predict(input)

        s_model = secure_model(model)
        s_output = s_model.predict(input)

        np.testing.assert_array_almost_equal(s_output, output, 4)


if __name__ == '__main__':
    unittest.main()
