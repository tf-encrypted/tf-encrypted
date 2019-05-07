"""Data processing helpers."""
#
# Based on:
# - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py
# - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py
#
import tensorflow as tf


def encode_image(value):
  """Encode images into a tf.train.Feature for a TFRecord."""
  bytes_list = tf.train.BytesList(value=[value.tostring()])
  return tf.train.Feature(bytes_list=bytes_list)


def decode_image(value):
  """Decode the image from a tf.train.Feature in a TFRecord."""
  image = tf.decode_raw(value, tf.uint8)
  image.set_shape((28 * 28))
  return image


def encode_label(value):
  """Encode a label into a tf.train.Feature for a TFRecord."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def decode_label(value):
  """Decode the label from a tf.train.Feature in a TFRecord."""
  return tf.cast(value, tf.int32)


def encode(image, label):
  """Encode an instance as a tf.train.Example for a TFRecord."""
  feature_dict = {'image': encode_image(image), 'label': encode_label(label)}
  features = tf.train.Features(feature=feature_dict)
  return tf.train.Example(features=features)


def decode(serialized_example):
  """Decode an instance from a tf.train.Example in a TFRecord."""
  features = tf.parse_single_example(serialized_example, features={
      'image': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.int64)
  })
  image = decode_image(features['image'])
  label = decode_label(features['label'])
  return image, label


def normalize(image, label):
  """Standardization of MNIST images."""
  x = tf.cast(image, tf.float32) / 255.
  image = (x - 0.1307) / 0.3081  # image = (x - mean) / std
  return image, label


def get_data_from_tfrecord(filename, batch_size: int):
  """Construct a TFRecordDataset iterator."""
  return tf.data.TFRecordDataset([filename]) \
                .map(decode) \
                .map(normalize) \
                .repeat() \
                .batch(batch_size) \
                .make_one_shot_iterator()
