import unittest

import time
import tensorflow as tf
import tensorflow_encrypted as tfe


class TestPond(unittest.TestCase):

    def setUp(self) -> None:
        tf.reset_default_graph()

    def test_pond_decode(self) -> None:

        with tfe.protocol.Pond() as prot:
            assert isinstance(prot, tfe.protocol.Pond)

            p_input = prot.define_private_input(
                tfe.io.InputProvider('input-provider', lambda: tf.constant([1.]))
            )
            pub_input = p_input.reveal()

            pond_public_tensor = pub_input
            int_100_tensor = pub_input.value_on_0
            tf_tensor = pond_public_tensor.prot._decode(int_100_tensor, pond_public_tensor.is_scaled)

            fetches = pub_input.value_on_0.backing

            with tfe.Session() as sess:
                for i in range(2):
                    decode_start = time.time()
                    fetches_out = sess.run(fetches)
                    decoder = pub_input
                    to_decode = decoder.value_on_0.from_decomposed(fetches_out)
                    decoder.prot._decode(to_decode, decoder.is_scaled)
                    total_decode_time = time.time() - decode_start

                    decode_tf_start = time.time()
                    sess.run(tf_tensor)
                    total_decode_tf_time = time.time() - decode_tf_start

                assert(abs(total_decode_time - total_decode_tf_time) < 0.01)


if __name__ == '__main__':
    unittest.main()
