import tensorflow as tf
import tf_encrypted as tfe

from tensorflow.keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.framework.graph_util_impl import remove_training_nodes


class PrivateModel():
    def __init__(self, output_node):
        self.output_node = output_node

    # TODO support multiple inputs
    def private_predict(self, input, input_name=None, tag="prediction"):
        if input_name is None:
            name = "private-input/api/0:0"
        else:
            name = input_name

        pl = tf.get_default_graph().get_tensor_by_name(name)

        with tfe.Session() as sess:
            sess.run(tf.global_variables_initializer())

            output = sess.run(
                self.output_node.reveal(),
                feed_dict={pl: input},
                tag=tag
            )

            return output


def load_graph(model_file, model_name=None):

    input_spec = []
    with gfile.GFile(model_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        for node in graph_def.node:
            if node.op != "Placeholder":
                continue

            input_spec.append({
                'name': node.name,
                'dtype': node.attr['dtype'].type,
                'shape': [1] + [int(d.size) for d in node.attr['shape'].shape.dim[1:]]
            })

    inputs = []
    for i, spec in enumerate(input_spec):
        def scope(i, spec):
            def provide_input() -> tf.Tensor:
                if model_name is None:
                    name = "api/{}".format(i)
                else:
                    name = "api/{}/{}".format(model_name, i)

                pl = tf.placeholder(tf.float32, shape=spec['shape'], name=name)
                return pl

            return provide_input

        inputs.append(scope(i, spec))

    return graph_def, inputs


def secure_model(model):
    session = K.get_session()
    min_graph = graph_util.convert_variables_to_constants(session, session.graph_def, [node.op.name for node in model.outputs])
    tf.train.write_graph(min_graph, '/tmp', 'model.pb', as_text=False)

    graph_def, inputs = load_graph('/tmp/model.pb')

    c = tfe.convert.convert.Converter()
    y = c.convert(remove_training_nodes(graph_def), tfe.convert.registry(), 'input-provider', inputs)

    return PrivateModel(y)
