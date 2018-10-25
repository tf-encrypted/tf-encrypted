import sys

import tensorflow as tf
import tf_encrypted as tfe
from tf_encrypted.layers import Conv2D, Dense, Sigmoid, Reshape

config = tfe.LocalConfig([
    'server0',
    'server1',
    'crypto-producer',
    'weights-provider',
    'prediction-client'
])

# config = tfe.RemoteConfig([
#     ('server0', 'localhost:4440'),
#     ('server1', 'localhost:4441'),
#     ('crypto-producer', 'localhost:4442'),
#     ('weights-provider', 'localhost:4443'),
#     ('prediction-client', 'localhost:4444')
# ])


if len(sys.argv) > 1:
    if isinstance(config, tfe.LocalConfig):
        raise Exception("You can launch a configured server only with a remote configuration")
    #
    # assume we're running as a server
    #

    player_name = str(sys.argv[1])

    server = config.server(player_name)
    server.start()
    server.join()
else:

    #
    # assume we're running as master
    #

    input_shape = [1, 3, 192, 192]
    conv11_fshape = [3, 3, 3, 64]
    conv12_fshape = [3, 3, 64, 64]
    pool1_shape = [1, 1, 64, 64]

    conv21_fshape = [3, 3, 64, 128]
    conv22_fshape = [3, 3, 128, 128]
    pool2_shape = [1, 1, 128, 128]

    conv31_fshape = [3, 3, 128, 256]
    conv32_fshape = [3, 3, 256, 256]
    conv33_fshape = [3, 3, 256, 256]
    pool3_shape = [1, 1, 256, 256]

    conv41_fshape = [3, 3, 256, 512]
    conv42_fshape = [3, 3, 512, 512]
    conv43_fshape = [3, 3, 512, 512]
    pool4_shape = [1, 1, 512, 512]

    conv51_fshape = [3, 3, 512, 512]
    conv52_fshape = [3, 3, 512, 512]
    conv53_fshape = [3, 3, 512, 512]
    pool5_shape = [1, 1, 512, 512]

    def provide_input_conv11weights() -> tf.Tensor:
        w = tf.random_normal(shape=conv11_fshape, dtype=tf.float32)
        return tf.Print(w, [w], message="w11:")

    def provide_input_conv12weights() -> tf.Tensor:
        w = tf.random_normal(shape=conv12_fshape, dtype=tf.float32)
        return tf.Print(w, [w], message="w12:")

    def provide_input_pool1weights() -> tf.Tensor:
        w = tf.random_normal(shape=pool1_shape, dtype=tf.float32)
        return tf.Print(w, [w], message="p1:")

    def provide_input_conv21weights() -> tf.Tensor:
        w = tf.random_normal(shape=conv21_fshape, dtype=tf.float32)
        return tf.Print(w, [w], message="w21:")

    def provide_input_conv22weights() -> tf.Tensor:
        w = tf.random_normal(shape=conv22_fshape, dtype=tf.float32)
        return tf.Print(w, [w], message="w22:")

    def provide_input_pool2weights() -> tf.Tensor:
        w = tf.random_normal(shape=pool2_shape, dtype=tf.float32)
        return tf.Print(w, [w], message="p2:")

    def provide_input_conv31weights() -> tf.Tensor:
        w = tf.random_normal(shape=conv31_fshape, dtype=tf.float32)
        return tf.Print(w, [w], message="w31:")

    def provide_input_conv32weights() -> tf.Tensor:
        w = tf.random_normal(shape=conv32_fshape, dtype=tf.float32)
        return tf.Print(w, [w], message="w32:")

    def provide_input_conv33weights() -> tf.Tensor:
        w = tf.random_normal(shape=conv33_fshape, dtype=tf.float32)
        return tf.Print(w, [w], message="w33:")

    def provide_input_pool3weights() -> tf.Tensor:
        w = tf.random_normal(shape=pool3_shape, dtype=tf.float32)
        return tf.Print(w, [w], message="p3:")

    def provide_input_conv41weights() -> tf.Tensor:
        w = tf.random_normal(shape=conv41_fshape, dtype=tf.float32)
        return tf.Print(w, [w], message="w41:")

    def provide_input_conv42weights() -> tf.Tensor:
        w = tf.random_normal(shape=conv42_fshape, dtype=tf.float32)
        return tf.Print(w, [w], message="w42:")

    def provide_input_conv43weights() -> tf.Tensor:
        w = tf.random_normal(shape=conv43_fshape, dtype=tf.float32)
        return tf.Print(w, [w], message="w43:")

    def provide_input_pool4weights() -> tf.Tensor:
        w = tf.random_normal(shape=pool4_shape, dtype=tf.float32)
        return tf.Print(w, [w], message="p4:")

    def provide_input_conv51weights() -> tf.Tensor:
        w = tf.random_normal(shape=conv51_fshape, dtype=tf.float32)
        return tf.Print(w, [w], message="w51:")

    def provide_input_conv52weights() -> tf.Tensor:
        w = tf.random_normal(shape=conv52_fshape, dtype=tf.float32)
        return tf.Print(w, [w], message="w52:")

    def provide_input_conv53weights() -> tf.Tensor:
        w = tf.random_normal(shape=conv53_fshape, dtype=tf.float32)
        return tf.Print(w, [w], message="w53:")

    def provide_input_pool5weights() -> tf.Tensor:
        w = tf.random_normal(shape=pool5_shape, dtype=tf.float32)
        return tf.Print(w, [w], message="p5:")

    def provide_input_prediction() -> tf.Tensor:
        x = tf.random_normal(shape=input_shape, dtype=tf.float32)
        return tf.Print(x, [x], message="x:")

    def receive_output(tensor: tf.Tensor) -> tf.Operation:
        return tf.Print(tensor, [tensor, tf.shape(tensor)], message="output:")

    with tfe.protocol.Pond(*config.get_players('server0, server1, crypto-producer')) as prot:

        print("Define the distributed graph")
        print("5 blocks of convolutions and a 2-layer FC")
        # load input for prediction
        x = prot.define_private_input('prediction-client', provide_input_prediction)

        print("Define Block 1")
        # Block 1
        conv11 = Conv2D(input_shape, conv11_fshape, 1, "SAME")
        initial_w_conv11 = prot.define_private_input('weights-provider', provide_input_conv11weights)
        conv11.initialize(initial_w_conv11)
        x = conv11.forward(x)
        x = Sigmoid(conv11.get_output_shape()).forward(x)
        conv12 = Conv2D(conv11.get_output_shape(), conv12_fshape, 1, "SAME")
        initial_w_conv12 = prot.define_private_input('weights-provider', provide_input_conv12weights)
        conv12.initialize(initial_w_conv12)
        x = conv12.forward(x)
        x = Sigmoid(conv12.get_output_shape()).forward(x)
        fake_pool1 = Conv2D(conv12.get_output_shape(), pool1_shape, 2, "SAME")
        initial_w_pool1 = prot.define_private_input('weights-provider', provide_input_pool1weights)
        fake_pool1.initialize(initial_w_pool1)
        x = fake_pool1.forward(x)

        print("Define Block 2")
        # Block 2
        conv21 = Conv2D(fake_pool1.get_output_shape(), conv21_fshape, 1, "SAME")
        initial_w_conv21 = prot.define_private_input('weights-provider', provide_input_conv21weights)
        conv21.initialize(initial_w_conv21)
        x = conv21.forward(x)
        x = Sigmoid(conv21.get_output_shape()).forward(x)
        conv22 = Conv2D(conv21.get_output_shape(), conv22_fshape, 1, "SAME")
        initial_w_conv22 = prot.define_private_input('weights-provider', provide_input_conv22weights)
        conv22.initialize(initial_w_conv22)
        x = conv22.forward(x)
        x = Sigmoid(conv22.get_output_shape()).forward(x)
        fake_pool2 = Conv2D(conv22.get_output_shape(), pool2_shape, 2, "SAME")
        initial_w_pool2 = prot.define_private_input('weights-provider', provide_input_pool2weights)
        fake_pool2.initialize(initial_w_pool2)
        x = fake_pool2.forward(x)

        print("Define Block 3")
        # Block 3
        conv31 = Conv2D(fake_pool2.get_output_shape(), conv31_fshape, 1, "SAME")
        initial_w_conv31 = prot.define_private_input('weights-provider', provide_input_conv31weights)
        conv31.initialize(initial_w_conv31)
        x = conv31.forward(x)
        x = Sigmoid(conv31.get_output_shape()).forward(x)
        conv32 = Conv2D(conv31.get_output_shape(), conv32_fshape, 1, "SAME")
        initial_w_conv32 = prot.define_private_input('weights-provider', provide_input_conv32weights)
        conv32.initialize(initial_w_conv32)
        x = conv32.forward(x)
        x = Sigmoid(conv32.get_output_shape()).forward(x)
        conv33 = Conv2D(conv32.get_output_shape(), conv33_fshape, 1, "SAME")
        initial_w_conv33 = prot.define_private_input('weights-provider', provide_input_conv33weights)
        conv33.initialize(initial_w_conv33)
        x = conv33.forward(x)
        x = Sigmoid(conv33.get_output_shape()).forward(x)
        fake_pool3 = Conv2D(conv33.get_output_shape(), pool3_shape, 2, "SAME")
        initial_w_pool3 = prot.define_private_input('weights-provider', provide_input_pool3weights)
        fake_pool3.initialize(initial_w_pool3)
        x = fake_pool3.forward(x)

        print("Define Block 4")
        # Block 4
        conv41 = Conv2D(fake_pool3.get_output_shape(), conv41_fshape, 1, "SAME")
        initial_w_conv41 = prot.define_private_input('weights-provider', provide_input_conv41weights)
        conv41.initialize(initial_w_conv41)
        x = conv41.forward(x)
        x = Sigmoid(conv41.get_output_shape()).forward(x)
        conv42 = Conv2D(conv41.get_output_shape(), conv42_fshape, 1, "SAME")
        initial_w_conv42 = prot.define_private_input('weights-provider', provide_input_conv42weights)
        conv42.initialize(initial_w_conv42)
        x = conv42.forward(x)
        x = Sigmoid(conv42.get_output_shape()).forward(x)
        conv43 = Conv2D(conv42.get_output_shape(), conv43_fshape, 1, "SAME")
        initial_w_conv43 = prot.define_private_input('weights-provider', provide_input_conv43weights)
        conv43.initialize(initial_w_conv43)
        x = conv43.forward(x)
        x = Sigmoid(conv43.get_output_shape()).forward(x)
        fake_pool4 = Conv2D(conv43.get_output_shape(), pool4_shape, 2, "SAME")
        initial_w_pool4 = prot.define_private_input('weights-provider', provide_input_pool4weights)
        fake_pool4.initialize(initial_w_pool4)
        x = fake_pool4.forward(x)

        print("Define Block 5")
        # Block 5
        conv51 = Conv2D(fake_pool4.get_output_shape(), conv51_fshape, 1, "SAME")
        initial_w_conv51 = prot.define_private_input('weights-provider', provide_input_conv51weights)
        conv51.initialize(initial_w_conv51)
        x = conv51.forward(x)
        x = Sigmoid(conv51.get_output_shape()).forward(x)
        conv52 = Conv2D(conv51.get_output_shape(), conv52_fshape, 1, "SAME")
        initial_w_conv52 = prot.define_private_input('weights-provider', provide_input_conv52weights)
        conv52.initialize(initial_w_conv52)
        x = conv52.forward(x)
        x = Sigmoid(conv52.get_output_shape()).forward(x)
        conv53 = Conv2D(conv52.get_output_shape(), conv53_fshape, 1, "SAME")
        initial_w_conv53 = prot.define_private_input('weights-provider', provide_input_conv53weights)
        conv53.initialize(initial_w_conv53)
        x = conv53.forward(x)
        x = Sigmoid(conv53.get_output_shape()).forward(x)
        fake_pool5 = Conv2D(conv53.get_output_shape(), pool5_shape, 2, "SAME")
        initial_w_pool5 = prot.define_private_input('weights-provider', provide_input_pool5weights)
        fake_pool5.initialize(initial_w_pool5)
        x = fake_pool5.forward(x)

        print("Define Reshape")
        reshape1 = Reshape(fake_pool5.get_output_shape(), [1, -1])
        x = reshape1.forward(x)

        print("Define 2-layer FC")
        dense1 = Dense(reshape1.get_output_shape(), 512)
        dense1.initialize()
        x = dense1.forward(x)
        x = Sigmoid(dense1.get_output_shape()).forward(x)
        dense2 = Dense(dense1.get_output_shape(), 2)
        dense2.initialize()
        y = dense2.forward(x)

        # send output
        prediction_op = prot.define_output(y, receive_output)

        with tfe.Session(config=config) as sess:
            print("Initialize tensors")
            sess.run(tf.global_variables_initializer(), tag='init')

            print("Predict")

            sess.run(prediction_op, tag='prediction')
