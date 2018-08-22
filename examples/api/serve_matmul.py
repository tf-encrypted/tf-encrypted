import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.python.platform import gfile

import tensorflow_encrypted as tfe
from tensorflow_encrypted.convert import convert
from tensorflow_encrypted.convert.register import register


# define the app
app = Flask(__name__)
CORS(app)  # needed for cross-domain requests, allow everything by default


def get_model_predict():

    model_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "matmul.pb"
    )

    config = tfe.LocalConfig([
        'server0',
        'server1',
        'crypto_producer',
        'weights_provider'
    ])

    tf.reset_default_graph()

    pl = tf.placeholder(tf.float32, shape=[1, 16])

    class PredictionInputProvider(tfe.io.InputProvider):
        def provide_input(self) -> tf.Tensor:
            return pl

    class PredictionOutputReceiver(tfe.io.OutputReceiver):
        def receive_output(self, tensor: tf.Tensor) -> tf.Tensor:
            return tf.Print(tensor, [tensor], message="tf.Print(output): ")

    with gfile.FastGFile(model_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    sess = config.session()
    with tfe.protocol.Pond(*config.get_players('server0, server1, crypto_producer')) as prot:
        input = PredictionInputProvider(config.get_player('master'))
        output = PredictionOutputReceiver(config.get_player('master'))

        c = convert.Converter(config, prot, config.get_player('weights_provider'))
        x = c.convert(graph_def, input, register())
        prediction_op = prot.define_output(x, output)

        tfe.run(sess, prot.initializer, tag='init')

    def model_api(input_data):
        output_data = x.reveal().eval(
            sess=sess,
            feed_dict={pl: input_data},
            tag='prediction',
        )
        return output_data[0].tolist()

    return model_api


predict = get_model_predict()


# API route
@app.route('/predict', methods=['POST'])
def api_predict():
    input_data = request.json
    app.logger.info("api_input: " + str(input_data))
    output_data = predict(input_data)
    app.logger.info("api_output: " + str(output_data))
    response = jsonify(output_data)
    return response


@app.route('/')
def index():
    return "Index API"


# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
