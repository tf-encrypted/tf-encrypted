"""Downloading the MNIST dataset and storing as TFRecords."""

import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from conv_convert import encode


def save_training_data(images, labels, filename):
  """Convert Keras MNIST data into TFRecords."""
  assert images.shape[0] == labels.shape[0]
  num_examples = images.shape[0]

  with tf.python_io.TFRecordWriter(filename) as writer:

    for index in range(num_examples):

      image = images[index]
      label = labels[index]
      example = encode(image, label)
      writer.write(example.SerializeToString())


(x_train, y_train), (x_test, y_test) = mnist.load_data()

data_dir = os.path.expanduser("./data/")
if not os.path.exists(data_dir):
  os.makedirs(data_dir)

save_training_data(x_train, y_train, os.path.join(data_dir, "train.tfrecord"))
save_training_data(x_test, y_test, os.path.join(data_dir, "test.tfrecord"))
