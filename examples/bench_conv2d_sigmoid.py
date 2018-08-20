import sys
import time
from typing import List

import tensorflow as tf
import tensorflow_encrypted as tfe
from tensorflow_encrypted.layers import Conv2D, Dense, Sigmoid, Reshape

config = tfe.LocalConfig([
    'server0',
    'server1',
    'crypto_producer',
    'weights_provider',
    'prediction_client'
])

# config = tfe.RemoteConfig([
#     ('server0', 'localhost:4440'),
#     ('server1', 'localhost:4441'),
#     ('crypto_producer', 'localhost:4442'),
#     ('weights_provider', 'localhost:4443'),
#     ('prediction_client', 'localhost:4444')
# ])


if len(sys.argv) > 1:
    if isinstance(config, tfe.LocalConfig):
        raise Exception("You can launch a configured server only with a remote configuration")
    #
    # assume we're running as a server
    #

    player_name: str = str(sys.argv[1])

    server = config.server(player_name)
    server.start()
    server.join()
else:

    #
    # assume we're running as master
    #

    conv11_fshape: List = [3, 3, 3, 64]
    conv12_fshape: List = [3, 3, 64, 64]
    pool1_shape: List = [1, 1, 64, 64]

    conv21_fshape: List = [3, 3, 64, 128]
    conv22_fshape: List = [3, 3, 128, 128]
    pool2_shape: List = [1, 1, 128, 128]

    conv31_fshape: List = [3, 3, 128, 256]
    conv32_fshape: List = [3, 3, 256, 256]
    conv33_fshape: List = [3, 3, 256, 256]
    pool3_shape: List = [1, 1, 256, 256]

    conv41_fshape: List = [3, 3, 256, 512]
    conv42_fshape: List = [3, 3, 512, 512]
    conv43_fshape: List = [3, 3, 512, 512]
    pool4_shape: List = [1, 1, 512, 512]

    conv51_fshape: List = [3, 3, 512, 512]
    conv52_fshape: List = [3, 3, 512, 512]
    conv53_fshape: List = [3, 3, 512, 512]
    pool5_shape: List = [1, 1, 512, 512]

    class Conv11WeightsInputProvider(tfe.io.InputProvider):
        def provide_input(self) -> tf.Tensor:
            w = tf.random_normal(shape=conv11_fshape, dtype=tf.float32)
            return tf.Print(w, [w], message="w11:")

    class Conv12WeightsInputProvider(tfe.io.InputProvider):
        def provide_input(self) -> tf.Tensor:
            w = tf.random_normal(shape=conv12_fshape, dtype=tf.float32)
            return tf.Print(w, [w], message="w12:")

    class Pool1WeightsInputProvider(tfe.io.InputProvider):
        def provide_input(self) -> tf.Tensor:
            w = tf.random_normal(shape=pool1_shape, dtype=tf.float32)
            return tf.Print(w, [w], message="p1:")

    class Conv21WeightsInputProvider(tfe.io.InputProvider):
        def provide_input(self) -> tf.Tensor:
            w = tf.random_normal(shape=conv21_fshape, dtype=tf.float32)
            return tf.Print(w, [w], message="w21:")

    class Conv22WeightsInputProvider(tfe.io.InputProvider):
        def provide_input(self) -> tf.Tensor:
            w = tf.random_normal(shape=conv22_fshape, dtype=tf.float32)
            return tf.Print(w, [w], message="w22:")

    class Pool2WeightsInputProvider(tfe.io.InputProvider):
        def provide_input(self) -> tf.Tensor:
            w = tf.random_normal(shape=pool2_shape, dtype=tf.float32)
            return tf.Print(w, [w], message="p2:")

    class Conv31WeightsInputProvider(tfe.io.InputProvider):
        def provide_input(self) -> tf.Tensor:
            w = tf.random_normal(shape=conv31_fshape, dtype=tf.float32)
            return tf.Print(w, [w], message="w31:")

    class Conv32WeightsInputProvider(tfe.io.InputProvider):
        def provide_input(self) -> tf.Tensor:
            w = tf.random_normal(shape=conv32_fshape, dtype=tf.float32)
            return tf.Print(w, [w], message="w32:")

    class Conv33WeightsInputProvider(tfe.io.InputProvider):
        def provide_input(self) -> tf.Tensor:
            w = tf.random_normal(shape=conv33_fshape, dtype=tf.float32)
            return tf.Print(w, [w], message="w33:")

    class Pool3WeightsInputProvider(tfe.io.InputProvider):
        def provide_input(self) -> tf.Tensor:
            w = tf.random_normal(shape=pool3_shape, dtype=tf.float32)
            return tf.Print(w, [w], message="p3:")

    class Conv41WeightsInputProvider(tfe.io.InputProvider):
        def provide_input(self) -> tf.Tensor:
            w = tf.random_normal(shape=conv41_fshape, dtype=tf.float32)
            return tf.Print(w, [w], message="w41:")

    class Conv42WeightsInputProvider(tfe.io.InputProvider):
        def provide_input(self) -> tf.Tensor:
            w = tf.random_normal(shape=conv42_fshape, dtype=tf.float32)
            return tf.Print(w, [w], message="w42:")

    class Conv43WeightsInputProvider(tfe.io.InputProvider):
        def provide_input(self) -> tf.Tensor:
            w = tf.random_normal(shape=conv43_fshape, dtype=tf.float32)
            return tf.Print(w, [w], message="w43:")

    class Pool4WeightsInputProvider(tfe.io.InputProvider):
        def provide_input(self) -> tf.Tensor:
            w = tf.random_normal(shape=pool4_shape, dtype=tf.float32)
            return tf.Print(w, [w], message="p4:")

    class Conv51WeightsInputProvider(tfe.io.InputProvider):
        def provide_input(self) -> tf.Tensor:
            w = tf.random_normal(shape=conv51_fshape, dtype=tf.float32)
            return tf.Print(w, [w], message="w51:")

    class Conv52WeightsInputProvider(tfe.io.InputProvider):
        def provide_input(self) -> tf.Tensor:
            w = tf.random_normal(shape=conv52_fshape, dtype=tf.float32)
            return tf.Print(w, [w], message="w52:")

    class Conv53WeightsInputProvider(tfe.io.InputProvider):
        def provide_input(self) -> tf.Tensor:
            w = tf.random_normal(shape=conv53_fshape, dtype=tf.float32)
            return tf.Print(w, [w], message="w53:")

    class Pool5WeightsInputProvider(tfe.io.InputProvider):
        def provide_input(self) -> tf.Tensor:
            w = tf.random_normal(shape=pool5_shape, dtype=tf.float32)
            return tf.Print(w, [w], message="p5:")

    class PredictionInputProvider(tfe.io.InputProvider):
        def provide_input(self) -> tf.Tensor:
            x = tf.random_normal(shape=[1, 3, 192, 192], dtype=tf.float32)
            return tf.Print(x, [x], message="x:")

    class PredictionOutputReceiver(tfe.io.OutputReceiver):
        def receive_output(self, tensor: tf.Tensor) -> tf.Operation:
            x = tf.Print(tensor, [tensor, tf.shape(tensor)], message="output:")
            return tf.group(x)

    weights_conv11 = Conv11WeightsInputProvider(config.get_player('weights_provider'))
    weights_conv12 = Conv12WeightsInputProvider(config.get_player('weights_provider'))
    weights_pool1 = Pool1WeightsInputProvider(config.get_player('weights_provider'))

    weights_conv21 = Conv21WeightsInputProvider(config.get_player('weights_provider'))
    weights_conv22 = Conv22WeightsInputProvider(config.get_player('weights_provider'))
    weights_pool2 = Pool2WeightsInputProvider(config.get_player('weights_provider'))

    weights_conv31 = Conv31WeightsInputProvider(config.get_player('weights_provider'))
    weights_conv32 = Conv32WeightsInputProvider(config.get_player('weights_provider'))
    weights_conv33 = Conv33WeightsInputProvider(config.get_player('weights_provider'))
    weights_pool3 = Pool3WeightsInputProvider(config.get_player('weights_provider'))

    weights_conv41 = Conv41WeightsInputProvider(config.get_player('weights_provider'))
    weights_conv42 = Conv42WeightsInputProvider(config.get_player('weights_provider'))
    weights_conv43 = Conv43WeightsInputProvider(config.get_player('weights_provider'))
    weights_pool4 = Pool4WeightsInputProvider(config.get_player('weights_provider'))

    weights_conv51 = Conv51WeightsInputProvider(config.get_player('weights_provider'))
    weights_conv52 = Conv52WeightsInputProvider(config.get_player('weights_provider'))
    weights_conv53 = Conv53WeightsInputProvider(config.get_player('weights_provider'))
    weights_pool5 = Pool5WeightsInputProvider(config.get_player('weights_provider'))

    prediction_input = PredictionInputProvider(config.get_player('prediction_client'))
    prediction_output = PredictionOutputReceiver(config.get_player('prediction_client'))

    with tfe.protocol.Pond(*config.get_players('server0, server1, crypto_producer')) as prot:

        print("Define the distributed graph")
        print("5 blocks of convolutions and a 2-layer FC")
        # load input for prediction
        x = prot.define_private_input(prediction_input)

        print("Define Block 1")
        # Block 1
        conv11 = Conv2D(conv11_fshape, 1, "SAME")
        initial_w_conv11 = prot.define_private_input(weights_conv11)
        conv11.initialize((1, 3, 192, 192), initial_w_conv11)
        x = conv11.forward(x)
        x = Sigmoid().forward(x)
        conv12 = Conv2D(conv12_fshape, 1, "SAME")
        initial_w_conv12 = prot.define_private_input(weights_conv12)
        conv12.initialize((1, 64, 192, 192), initial_w_conv12)
        x = conv12.forward(x)
        x = Sigmoid().forward(x)
        fake_pool1 = Conv2D(pool1_shape, 2, "SAME")
        initial_w_pool1 = prot.define_private_input(weights_pool1)
        fake_pool1.initialize((1, 64, 192, 192), initial_w_pool1)
        x = fake_pool1.forward(x)

        print("Define Block 2")
        # Block 2
        conv21 = Conv2D(conv21_fshape, 1, "SAME")
        initial_w_conv21 = prot.define_private_input(weights_conv21)
        conv21.initialize((1, 64, 96, 96), initial_w_conv21)
        x = conv21.forward(x)
        x = Sigmoid().forward(x)
        conv22 = Conv2D(conv22_fshape, 1, "SAME")
        initial_w_conv22 = prot.define_private_input(weights_conv22)
        conv22.initialize((1, 128, 96, 96), initial_w_conv22)
        x = conv22.forward(x)
        x = Sigmoid().forward(x)
        fake_pool2 = Conv2D(pool2_shape, 2, "SAME")
        initial_w_pool2 = prot.define_private_input(weights_pool2)
        fake_pool2.initialize((1, 128, 96, 96), initial_w_pool2)
        x = fake_pool2.forward(x)

        print("Define Block 3")
        # Block 3
        conv31 = Conv2D(conv31_fshape, 1, "SAME")
        initial_w_conv31 = prot.define_private_input(weights_conv31)
        conv31.initialize((1, 128, 48, 48), initial_w_conv31)
        x = conv31.forward(x)
        x = Sigmoid().forward(x)
        conv32 = Conv2D(conv32_fshape, 1, "SAME")
        initial_w_conv32 = prot.define_private_input(weights_conv32)
        conv32.initialize((1, 256, 48, 48), initial_w_conv32)
        x = conv32.forward(x)
        x = Sigmoid().forward(x)
        conv33 = Conv2D(conv33_fshape, 1, "SAME")
        initial_w_conv33 = prot.define_private_input(weights_conv33)
        conv33.initialize((1, 256, 48, 48), initial_w_conv33)
        x = conv33.forward(x)
        x = Sigmoid().forward(x)
        fake_pool3 = Conv2D(pool3_shape, 2, "SAME")
        initial_w_pool3 = prot.define_private_input(weights_pool3)
        fake_pool3.initialize((1, 256, 48, 48), initial_w_pool3)
        x = fake_pool3.forward(x)

        print("Define Block 4")
        # Block 4
        conv41 = Conv2D(conv41_fshape, 1, "SAME")
        initial_w_conv41 = prot.define_private_input(weights_conv41)
        conv41.initialize((1, 256, 24, 24), initial_w_conv41)
        x = conv41.forward(x)
        x = Sigmoid().forward(x)
        conv42 = Conv2D(conv42_fshape, 1, "SAME")
        initial_w_conv42 = prot.define_private_input(weights_conv42)
        conv42.initialize((1, 512, 24, 24), initial_w_conv42)
        x = conv42.forward(x)
        x = Sigmoid().forward(x)
        conv43 = Conv2D(conv43_fshape, 1, "SAME")
        initial_w_conv43 = prot.define_private_input(weights_conv43)
        conv43.initialize((1, 512, 24, 24), initial_w_conv43)
        x = conv43.forward(x)
        x = Sigmoid().forward(x)
        fake_pool4 = Conv2D(pool4_shape, 2, "SAME")
        initial_w_pool4 = prot.define_private_input(weights_pool4)
        fake_pool4.initialize((1, 512, 24, 24), initial_w_pool4)
        x = fake_pool4.forward(x)

        print("Define Block 5")
        # Block 5
        conv51 = Conv2D(conv51_fshape, 1, "SAME")
        initial_w_conv51 = prot.define_private_input(weights_conv51)
        conv51.initialize((1, 512, 12, 12), initial_w_conv51)
        x = conv51.forward(x)
        x = Sigmoid().forward(x)
        conv52 = Conv2D(conv52_fshape, 1, "SAME")
        initial_w_conv52 = prot.define_private_input(weights_conv52)
        conv52.initialize((1, 512, 12, 12), initial_w_conv52)
        x = conv52.forward(x)
        x = Sigmoid().forward(x)
        conv53 = Conv2D(conv53_fshape, 1, "SAME")
        initial_w_conv53 = prot.define_private_input(weights_conv53)
        conv53.initialize((1, 512, 12, 12), initial_w_conv53)
        x = conv53.forward(x)
        x = Sigmoid().forward(x)
        fake_pool5 = Conv2D(pool5_shape, 2, "SAME")
        initial_w_pool5 = prot.define_private_input(weights_pool5)
        fake_pool5.initialize((1, 512, 12, 12), initial_w_pool5)
        x = fake_pool5.forward(x)

        print("Define Reshape")
        x = Reshape(shape=[1, -1]).forward(x)

        print("Define 2-layer FC")
        dense1 = Dense(512*6*6, 512)
        dense1.initialize()
        x = dense1.forward(x)
        x = Sigmoid().forward(x)
        dense2 = Dense(512, 2)
        dense2.initialize()
        y = dense2.forward(x)

        # send output
        prediction_op = prot.define_output(y, prediction_output)

        with config.session() as sess:
            print("Initialize tensors")
            tfe.run(sess, tf.global_variables_initializer(), tag='init')

            print("Predict")

            start = time.time()
            for i in range(1):
                tfe.run(sess, prediction_op, tag='prediction')
            end = time.time()
            print((end - start)/5)
