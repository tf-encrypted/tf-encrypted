import os
import uuid
from threading import Thread
from typing import Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.python.platform import gfile

import tensorflow_encrypted as tfe
from tensorflow_encrypted.convert import convert
from tensorflow_encrypted.convert.register import register

_global_memory: Dict[str, Any] = {}

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

    class PredictionInputProvider(tfe.io.InputProvider):
        def provide_input(self) -> tf.Tensor:
            return tf.placeholder(tf.float32, shape=[1, 16], name="api")

    class PredictionOutputReceiver(tfe.io.OutputReceiver):
        def receive_output(self, tensor: tf.Tensor) -> tf.Tensor:
            return tf.Print(tensor, [tensor], message="tf.Print(output): ")

    with gfile.FastGFile(model_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    sess = config.session()
    with tfe.protocol.Pond(*config.get_players('server0, server1, crypto_producer')) as prot:
        # The machine which feeds input must be the flask server for now
        input = PredictionInputProvider(config.get_player('master'))

        c = convert.Converter(config, prot, config.get_player('weights_provider'))
        x = c.convert(graph_def, input, register())

        # Not sure how to use this scheme for now
        # output = PredictionOutputReceiver(config.get_player('master'))
        # prediction_op = prot.define_output(x, output)

        tfe.run(sess, prot.initializer, tag='init')

    pl = tf.get_default_graph().get_tensor_by_name("private-input/api:0")

    def predict_fn(request_id: str, input_data: Any) -> None:
        output_data = x.reveal().eval(
            sess=sess,
            feed_dict={pl: input_data},
            tag='prediction',
        )

        _global_memory[request_id] = output_data[0].tolist()

    return predict_fn


predict = get_model_predict()


# API route
@app.route('/predict', methods=['POST'])
def api_predict():
    input_data = request.json
    app.logger.info("api_predict_input: " + str(input_data))

    request_id = str(uuid.uuid4())

    thread = Thread(target=predict, kwargs={
        'request_id': request_id,
        'input_data': input_data
    })
    thread.start()

    app.logger.info("api_predict_output: " + request_id)
    data = {'request_id': request_id}
    response = jsonify(data)
    return response


@app.route('/poll', methods=['POST'])
def api_poll():
    request_id = request.json
    app.logger.info("api_poll_input: " + str(request_id))
    print(_global_memory)

    output_data = _global_memory.pop(request_id, None)

    app.logger.info("api_poll_output: " + str(output_data))
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
