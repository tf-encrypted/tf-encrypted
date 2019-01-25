import tensorflow as tf
import tf_encrypted as tfe


class PrivateModel():
    def __init__(self, output_node):
        self.output_node = output_node

    # TODO support multiple inputs
    def predict(self, input):
        name = "private-input/api/0:0"
        pl = tf.get_default_graph().get_tensor_by_name(name)

        with tfe.Session() as sess:
            sess.run(tf.global_variables_initializer())

            output = sess.run(
                self.output_node.reveal(),
                feed_dict={pl: input},
                tag='prediction'
            )

            return output
