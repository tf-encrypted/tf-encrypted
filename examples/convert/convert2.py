import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.platform import gfile
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
import _pickle as pickle
import numpy as np
import tf_encrypted as tfe
from tf_encrypted.convert import convert
from tf_encrypted.convert.register import registry
import time
import sys
from tf_encrypted.performance import Performance
import os
from tf_encrypted.utils import print_banner


if len(sys.argv) > 1:
    # config file was specified
    config_file = sys.argv[1]
    config = tfe.RemoteConfig.load(config_file)
    tfe.set_config(config)
else:
    # Always best practice to preset all players to avoid invalid device errors
    config = tfe.LocalConfig(player_names=["server0", "server1", "server2", "prediction-client", "weights-provider"])
    tfe.set_config(config)

check_nodes = ["conv2_block3_preact_bn/FusedBatchNormV3", "conv2_block3_1_bn/FusedBatchNormV3"]

directory = os.path.dirname(os.path.abspath(__file__))

def load_images(preprocess=True):
    with open(os.path.join(directory, "n02109961_36_enc.pkl"), "rb") as ff:
        images = np.array(pickle.load(ff))

    if preprocess:
        images = preprocess_input(images)

    numImages = len(images)
    return images

def export_resnet50():
    print_banner("Export Resnet50 Model")
    images = load_images()
    tf.keras.backend.set_learning_phase(0)
    tf.keras.backend.set_image_data_format('channels_last')
    tf.keras.backend.set_floatx('float32')

    output_graph_filename = os.path.join(directory, "resnet50.pb")

    with tf.Session() as sess:
        # Must construct the model inside this session, otherwise it might be complicated to freeze the graph later due to unitialized variables
        model = tf.keras.applications.ResNet50V2(weights='imagenet')
        preds = model.predict(images)
        print('Predicted:', decode_predictions(preds, top=10)[0])

        model_output = model.output.name.replace(':0', '')
        constant_graph = graph_util.convert_variables_to_constants(
            sess, sess.graph.as_graph_def(), [model_output]
        )
        frozen_graph = graph_util.remove_training_nodes(
            constant_graph
        )
        with open(output_graph_filename, 'wb') as f:
            f.write(frozen_graph.SerializeToString())


def load_frozen_resnet50():
    print_banner("Load Frozen Resnet50")
    model_filename = os.path.join(directory, "resnet50.pb")
    with gfile.GFile(model_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    return graph_def


def verify_frozen_resnet50():
    print_banner("Verify Frozen Resnet50")
    with tf.Graph().as_default():
        graph_def = load_frozen_resnet50()
        importer.import_graph_def(graph_def, name="")
        images = load_images()
        with tf.Session() as sess:
            input_node = sess.graph.get_tensor_by_name("input_1:0")
            output_node = sess.graph.get_tensor_by_name("probs/Softmax:0")
            preds = sess.run(output_node, feed_dict={input_node: images})
            print('Predicted:', decode_predictions(preds, top=10)[0])

            # out_tensors = [sess.graph.get_tensor_by_name(check+":0") for check in check_nodes]
            # out_tensors = sess.run(out_tensors, feed_dict={input_node: images})
            # for i in range(len(check_nodes)):
                # print(check_nodes[i], ": \n", out_tensors[i])


def convert_to_tfe_model(graph_def):
    print_banner("Convert Plain Resnet50 to TFE Resnet50")

    def provide_input() -> tf.Tensor:
        images = load_images()
        return tf.constant(images)


    def receive_output(tensor: tf.Tensor) -> tf.Tensor:
        tf.print(tensor, [tensor])
        return tensor


    with tfe.protocol.ABY3() as prot:

        c = convert.Converter(registry(), config=config, protocol=prot, model_provider=config.get_player("weights-provider"))
        x = c.convert(
            graph_def, config.get_player("prediction-client"), provide_input
        )

        with tfe.Session(config=config) as sess:
            sess.run(tfe.global_variables_initializer(), tag="init")

            Performance.time_log("Resnet50 Prediction 1st run")
            preds = sess.run(x.reveal(), tag="prediction")
            Performance.time_log("Resnet50 Prediction 1st run")

            Performance.time_log("Resnet50 Prediction 2nd run")
            preds = sess.run(x.reveal(), tag="prediction")
            Performance.time_log("Resnet50 Prediction 2nd run")

            print('Predicted:', decode_predictions(preds, top=10)[0])

            # out_tensors = [c.outputs[check].reveal() for check in check_nodes]
            # out_tensors = sess.run(out_tensors)
            # for i in range(len(check_nodes)):
                # print(check_nodes[i], ": \n", out_tensors[i])



# load_images()
export_resnet50()
verify_frozen_resnet50()

graph_def = load_frozen_resnet50()
convert_to_tfe_model(graph_def)
