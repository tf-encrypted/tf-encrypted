
#
# Based on:
# - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py
# - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py
#
from typing import Tuple
import tensorflow as tf


def encode_image(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tostring()]))


def decode_image(value):
    image = tf.decode_raw(value, tf.uint8)
    image.set_shape((28 * 28))
    return image


def encode_label(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def decode_label(value):
    return tf.cast(value, tf.int32)


def encode(image, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'image': encode_image(image),
        'label': encode_label(label)
    }))


def decode(serialized_example):
    features = tf.parse_single_example(serialized_example, features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })
    image = decode_image(features['image'])
    label = decode_label(features['label'])
    return image, label


def normalize(image, label):
    x = tf.cast(image, tf.float32) / 255.
    image = (x - 0.1307) / 0.3081  # image = (x - mean) / std
    return image, label


def get_data_from_tfrecord(filename: str, bs: int) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.data.TFRecordDataset([filename]) \
                  .map(decode) \
                  .map(normalize) \
                  .repeat() \
                  .batch(bs) \
                  .make_one_shot_iterator()
