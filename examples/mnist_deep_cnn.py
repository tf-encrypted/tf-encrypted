
# This code is based on the following example:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_deep.py

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

import numpy

from tensorflow.python.framework import graph_io
from tensorflow.python.framework import graph_util

FLAGS = None


def deepnn(x):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
    Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9).
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 16 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 16])
        b_conv1 = bias_variable([16])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = avg_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 16 feature maps to 16.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 16, 16])
        b_conv2 = bias_variable([16])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = avg_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 256 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([256, 100])
        b_fc1 = bias_variable([100])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Map the 256 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([100, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    return y_conv


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def avg_pool_2x2(x):
    """avg_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.int64, [None])

    # Build the graph for the deep net
    y_conv = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            labels=y_, logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1]})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1]})

        # compute in batches to avoid OOM on GPUs
        accuracy_l = []
        for _ in range(20):
            batch = mnist.test.next_batch(500, shuffle=False)

            accuracy_l.append(accuracy.eval(feed_dict={x: batch[0], y_: batch[1]}))

        print('test accuracy %g' % numpy.mean(accuracy_l))

        # Export model to protobuf file
        current_dir = os.getcwd()
        pb_filename = '/test_data/mnist_model.pb'
        export_to_pb(sess, y_conv, current_dir + pb_filename)

        # Export sample mnist data
        np_filename = '/test_data/mnist_input.npy'
        numpy.save(current_dir + np_filename, mnist.test.images[0])


def export_to_pb(sess, x, filename):
    pred_names = ['output']
    tf.identity(x, name=pred_node_names[0])

    graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_names)

    graph = graph_util.remove_training_nodes(graph)
    path = graph_io.write_graph(graph, ".", filename, as_text=False)
    print('saved the frozen graph (ready for inference) at: ', filename)

    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
