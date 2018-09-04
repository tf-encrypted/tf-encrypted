import os
import numpy as np
import tensorflow as tf
import tensorflow.saved_model as sm

import tensorflow_encrypted as tfe

path_index = 1
export_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'logreg',
    str(path_index)
)
while os.path.exists(export_path):
    path_index += 1
    export_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'logreg',
        str(path_index)
    )

# config = tfe.LocalConfig([
#     'server0',
#     'server1',
#     'crypto_producer',
#     'weights_provider',
#     'prediction_client',
# ])
config = tfe.RemoteConfig({
    'server0': '0.0.0.0:4440',
    'server1': '0.0.0.0:4441',
    'crypto_producer': '0.0.0.0:4442',
    'weights_provider': '0.0.0.0:4443',
    'prediction_client': '0.0.0.0:4444',
})

np_input = np.array([.1, -.1, .2, -.2]).reshape(2, 2)
np_w = np.array([.1, .2, .3, .4]).reshape(2, 2)
np_b = np.array([.1, .2, .3, .4]).reshape(2, 2)


class WeightsInputProvider(tfe.io.InputProvider):

        def provide_input(self) -> tf.Tensor:
            w = tf.constant(np_w)
            return tf.Print(w, [w], message='W:')


class BiasInputProvider(tfe.io.InputProvider):

        def provide_input(self) -> tf.Tensor:
            b = tf.constant(np_b)
            return tf.Print(b, [b], message='b:')


class PredictionInputProvider(tfe.io.InputProvider):

    def provide_input(self) -> tf.Tensor:
        x = tf.constant(np_input, name="x_input")
        return tf.Print(x, [x], message='x:')

    def get_input_tensor(self) -> tf.Tensor:
        # TODO: Make this more robust
        return tf.get_default_graph().get_tensor_by_name("private-input/x_input:0")


prediction_input = PredictionInputProvider(config.get_player('prediction_client'))

weights_input = WeightsInputProvider(config.get_player('weights_provider'))
bias_input = BiasInputProvider(config.get_player('weights_provider'))

with tfe.protocol.Pond(*config.get_players('server0, server1, crypto_producer')) as prot:

    # input
    x = prot.define_private_input(prediction_input)

    # parameters
    initial_w = prot.define_private_input(weights_input)
    w = prot.define_private_variable(initial_w)
    initial_b = prot.define_private_input(bias_input)
    b = prot.define_private_variable(initial_b)

    # prediction
    y = prot.sigmoid(w.dot(x) + b).reveal()


with config.session() as sess:
    tfe.run(sess, tf.global_variables_initializer(), tag='init')

    prediction = y.eval(sess, tag='reveal')
    print('tf pred: ', prediction)

    builder = sm.builder.SavedModelBuilder(export_path)

    # Build the signature_def_map.
    # TODO: Find a way to pass the 2 shares secured representation as inputs
    # potentially as bytes/string?
    input_tensor = prediction_input.get_input_tensor()
    tensor_info_x = sm.utils.build_tensor_info(input_tensor)
    # TODO: Find a way to decode using Tensorflow operations to return the
    # 2 output shares
    tensor_info_y = sm.utils.build_tensor_info(y.value_on_0.backing[0])

    # TODO: create 2 new signatures:
    #  REGRESS_INPUTS_SECURE
    #  REGRESS_OUTPUTS_SECURE
    regress_signature = (
        sm.signature_def_utils.build_signature_def(
            inputs={
                sm.signature_constants.REGRESS_INPUTS:
                    tensor_info_x
            },
            outputs={
                sm.signature_constants.REGRESS_OUTPUTS:
                    tensor_info_y
            },
            method_name=sm.signature_constants.REGRESS_METHOD_NAME
        )
    )

    prediction_signature = (
        sm.signature_def_utils.build_signature_def(
            inputs={'data': tensor_info_x},
            outputs={'reg_out': tensor_info_y},
            method_name=sm.signature_constants.PREDICT_METHOD_NAME
        )
    )

    builder.add_meta_graph_and_variables(
        sess, [sm.tag_constants.SERVING],
        signature_def_map={
            'predict_reg':
                prediction_signature,
            sm.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                regress_signature,
        },
        main_op=tf.tables_initializer(),
        strip_default_attrs=True,
        clear_devices=False,
    )

    print('Saving to %s' % export_path)
    builder.save(as_text=True)
