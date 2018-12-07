import unittest
import os
import logging

from typing import List, Tuple

import numpy as np
import tensorflow as tf
import tf_encrypted as tfe
from tf_encrypted.convert import Converter
from tf_encrypted.convert.register import register

from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io


global_filename = ''

class TestConvert(unittest.TestCase):
    def test_required_space_to_batch_paddings_convert(self):

        print("Required Space")
        tf.reset_default_graph()

        global global_filename
        global_filename = "required_space_to_batch_paddings.pb"

        input_shape = [3]

        path = export_required_space_to_batch_paddings(global_filename, input_shape)

        tf.reset_default_graph()

        graph_def = read_graph(path)

        tf.reset_default_graph()

        pad_tf, crop_tf = run_required_space_to_batch_paddings(input_shape)

        tf.reset_default_graph()

        with tfe.protocol.Pond() as prot:
            prot.clear_initializers()

            def provide_input():
                return tf.constant(np.array([4, 1, 3]), dtype=tf.int32)

            converter = Converter(tfe.get_config(), prot, 'model-provider')

            x = converter.convert(graph_def, register(), 'input-provider', provide_input)

            with tfe.Session() as sess:
                sess.run(tf.global_variables_initializer())

                pad_tfe, crop_tfe = sess.run(x, tag='reveal')

        np.testing.assert_array_almost_equal(pad_tfe, pad_tf, decimal=3)
        np.testing.assert_array_almost_equal(crop_tfe, crop_tf, decimal=3)


def run_required_space_to_batch_paddings(input_shape: List[int]):

    x = tf.placeholder(tf.int32, shape=input_shape, name="input")
    y = tf.constant(np.array([2, 3, 2]), dtype=tf.int32)
    p = tf.constant(np.array([[2,3],[4,3],[5,2]]), dtype=tf.int32)

    out = tf.required_space_to_batch_paddings(x, y, base_paddings=p)

    with tf.Session() as sess:
        output = sess.run(out, feed_dict={x: np.array([4, 1, 3])})
        tf.summary.FileWriter('output/', sess.graph)

    return output


def export_required_space_to_batch_paddings(filename: str, input_shape: List[int]):

    x = tf.placeholder(tf.int32, shape=input_shape, name="input")
    y = tf.constant(np.array([2, 3, 2]), dtype=tf.int32)
    p = tf.constant(np.array([[2,3],[4,3],[5,2]]), dtype=tf.int32)

    out = tf.required_space_to_batch_paddings(x, y, base_paddings=p)

    return export(out, filename)


def export(x: tf.Tensor, filename: str):
    with tf.Session() as sess:
        pred_node_names = ["output"]
        tf.identity(x, name=pred_node_names[0])

        graph = graph_util.convert_variables_to_constants(sess,
                                                          sess.graph.as_graph_def(),
                                                          pred_node_names)

        graph = graph_util.remove_training_nodes(graph)

        path = graph_io.write_graph(graph, ".", filename, as_text=False)

    return path


def read_graph(path: str):
    with gfile.FastGFile(path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    return graph_def


if __name__ == '__main__':
    unittest.main()