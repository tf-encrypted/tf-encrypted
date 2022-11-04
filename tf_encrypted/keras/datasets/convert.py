"""Data processing helpers."""
#
# Based on:
# - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py  # noqa
# - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py  # noqa
#
import tensorflow as tf


def encode_image(image):
    """Encode an image as a tf.train.Example for a TFRecord."""
    encoded_image = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[image.tostring()])
    )
    feature_dict = {"image": encoded_image}
    features = tf.train.Features(feature=feature_dict)
    return tf.train.Example(features=features)


def decode_image(serialized_example):
    """Decode an image from a tf.train.Example in a TFRecord."""
    features = tf.io.parse_single_example(
        serialized_example,
        features={"image": tf.io.FixedLenFeature([], tf.string)},
    )
    image = tf.io.decode_raw(features["image"], tf.uint8)
    return image


def encode_label(label):
    """Encode a label as a tf.train.Example for a TFRecord."""
    encoded_label = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    feature_dict = {"label": encoded_label}
    features = tf.train.Features(feature=feature_dict)
    return tf.train.Example(features=features)


def decode_label(serialized_example):
    """Decode a label from a tf.train.Example in a TFRecord."""
    features = tf.io.parse_single_example(
        serialized_example,
        features={"label": tf.io.FixedLenFeature([], tf.int64)},
    )
    label = tf.cast(features["label"], tf.int32)
    return label


def encode_data(image, label):
    """Encode both image and label as a tf.train.Example for a TFRecord."""
    encoded_image = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[image.tostring()])
    )
    encoded_label = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    feature_dict = {"image": encoded_image, "label": encoded_label}
    features = tf.train.Features(feature=feature_dict)
    return tf.train.Example(features=features)


def decode_data(serialized_example):
    """Decode both image and label from a tf.train.Example in a TFRecord."""
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
        },
    )
    image = tf.io.decode_raw(features["image"], tf.uint8)
    label = tf.cast(features["label"], tf.int32)
    return image, label


def normalize(image, label):
    """Standardization of MNIST images."""
    x = tf.cast(image, tf.float32) / 255.0
    image = (x - 0.1307) / 0.3081  # image = (x - mean) / std
    return image, label


# def get_data_from_tfrecord(filename, batch_size: int):
#     """Construct a TFRecordDataset iterator."""
#     return (
#         tf.data.TFRecordDataset([filename])
#         .map(decode)
#         .map(normalize)
#         .repeat()
#         .batch(batch_size)
#     )


def save_data(images, labels, filename):
    """Convert both image and label data into TFRecords."""
    assert images.shape[0] == labels.shape[0]
    num_examples = images.shape[0]

    with tf.io.TFRecordWriter(filename) as writer:

        for index in range(num_examples):

            image = images[index]
            label = labels[index]
            example = encode_data(image, label)
            writer.write(example.SerializeToString())


def save_image(images, filename):
    """Convert image data into TFRecords."""
    num_examples = images.shape[0]

    with tf.io.TFRecordWriter(filename) as writer:

        for index in range(num_examples):

            image = images[index]
            example = encode_image(image)
            writer.write(example.SerializeToString())


def save_label(labels, filename):
    """Convert label data into TFRecords."""
    num_examples = labels.shape[0]

    with tf.io.TFRecordWriter(filename) as writer:

        for index in range(num_examples):

            label = labels[index]
            example = encode_label(label)
            writer.write(example.SerializeToString())
