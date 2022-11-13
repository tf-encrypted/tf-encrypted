import tensorflow as tf

from tf_encrypted.keras.datasets import MnistDataset
from tf_encrypted.keras.datasets.convert import decode_data
from tf_encrypted.keras.datasets.convert import decode_image
from tf_encrypted.keras.datasets.convert import decode_label


class LRMnistDataset(MnistDataset):
    """
    Use Mnist Dataset for LR training,
    small digits (0-4) and large digits (5-9) are calssified as two class
    """

    def __init__(self, batch_size=32, train=True) -> None:
        super().__init__(batch_size, train)
        self.num_classes = 2

    def _image_generator(self):
        def normalize(image):
            image = tf.cast(image, tf.float64) / 255.0
            return image

        def shaping(image):
            image = tf.reshape(
                image,
                shape=[self.batch_size, self.row, self.column, self.channel],
            )
            return image

        dataset = tf.data.TFRecordDataset([self.file_name])
        dataset = (
            dataset.map(decode_image)
            .map(normalize)
            .cache()
            .shuffle(self.num_samples, seed=11111)
            .batch(self.batch_size, drop_remainder=True)
            .map(shaping)
        )  # drop remainder because we need to fix batch size in private model
        if self.train:
            dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        image_iter = iter(dataset)

        return image_iter

    def _label_generator(self):
        def normalize(label):
            label = tf.cast(tf.math.greater(label, 4), dtype=tf.float64)
            return label

        def shaping(label):
            label = tf.reshape(label, [self.batch_size, 1])
            return label

        dataset = tf.data.TFRecordDataset([self.file_name])
        dataset = (
            dataset.map(decode_label)
            .map(normalize)
            .cache()
            .shuffle(self.num_samples, seed=11111)
            .batch(self.batch_size, drop_remainder=True)
            .map(shaping)
        )  # drop remainder because we need to fix batch size in private model
        if self.train:
            dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        label_iter = iter(dataset)

        return label_iter

    def _data_generator(self):
        def normalize(image, label):
            image = tf.cast(image, tf.float64) / 255.0
            label = tf.cast(tf.math.greater(label, 4), dtype=tf.float64)
            return image, label

        def shaping(image, label):
            image = tf.reshape(
                image,
                shape=[self.batch_size, self.row, self.column, self.channel],
            )
            label = tf.reshape(label, [self.batch_size, 1])
            return image, label

        dataset = tf.data.TFRecordDataset([self.file_name])
        dataset = (
            dataset.map(decode_data)
            .map(normalize)
            .cache()
            .shuffle(self.num_samples, seed=11111)
            .batch(self.batch_size, drop_remainder=True)
            .map(shaping)
        )  # drop remainder because we need to fix batch size in private model
        if self.train:
            dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        data_iter = iter(dataset)

        return data_iter
