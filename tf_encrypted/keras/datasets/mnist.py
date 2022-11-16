import os

import tensorflow as tf
from tensorflow.keras.datasets import mnist

from .convert import decode_data
from .convert import decode_image
from .convert import decode_label
from .convert import save_data
from .convert import save_image
from .convert import save_label


class MnistDataset:
    """
    Save mnist data into TFRecordDataset and get samples from generator
    """

    def __init__(self, batch_size=32, train=True) -> None:
        self.batch_size = batch_size
        self.train = train
        if self.train:
            self.num_samples = 60000
        else:
            self.num_samples = 10000
        self.row = 28
        self.column = 28
        self.channel = 1
        self.num_classes = 10
        self.iterations = self.num_samples // self.batch_size
        self.file_name = None

    def __getitem__(self, slc):
        if isinstance(slc, slice):
            return HMnistDataset(slc, self.batch_size, self.train)
        if isinstance(slc, tuple) and slc[0] == slice(None, None, None):
            return VMnistDataset(slc, self.batch_size, self.train)
        raise IndexError("Only support split dataset horizontally or vertically")

    def reset_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.iterations = self.num_samples // self.batch_size

    @property
    def batch_shape(self):
        return [self.batch_size, self.row, self.column, self.channel]

    def _data_generator(self):
        def normalize(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            label = tf.one_hot(label, 10)
            return image, label

        def shaping(image, label):
            image = tf.reshape(
                image,
                shape=[self.batch_size, self.row, self.column, self.channel],
            )
            label = tf.reshape(label, shape=[self.batch_size, self.num_classes])
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

    def _image_generator(self):
        def normalize(image):
            image = tf.cast(image, tf.float32) / 255.0
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
            label = tf.one_hot(label, 10)
            return label

        def shaping(label):
            label = tf.reshape(label, shape=[self.batch_size, self.num_classes])
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

    def generator_builder(self, image=True, label=True):
        if not image and not label:
            return None

        directory = os.path.join(os.getcwd(), "data")
        if not os.path.exists(directory):
            os.mkdir(directory)

        if self.file_name is None:
            self.file_name = os.path.join(directory, "mnist")
            if self.train:
                self.file_name += "_train"
                (x, y), (_, _) = mnist.load_data()
            else:
                self.file_name += "_test"
                (_, _), (x, y) = mnist.load_data()
            if image:
                self.file_name += "_image"
            if label:
                self.file_name += "_label"
            self.file_name += ".tfrecord"

        if image:
            if label:
                if not os.path.exists(self.file_name):
                    save_data(x, y, self.file_name)
                return self._data_generator
            else:
                if not os.path.exists(self.file_name):
                    save_image(x, self.file_name)
                return self._image_generator
        else:
            if not os.path.exists(self.file_name):
                save_label(y, self.file_name)
            return self._label_generator


class HMnistDataset(MnistDataset):
    """
    Horizontally splited mnist dataset
    """

    def __init__(self, slc, batch_size=32, train=True) -> None:
        super().__init__(batch_size, train)
        self.slc = slc
        self.num_samples = self.slc.stop - self.slc.start
        self.iterations = self.num_samples // self.batch_size

    def generator_builder(self, image=True, label=True):
        if not image and not label:
            return None

        directory = os.path.join(os.getcwd(), "data")
        if not os.path.exists(directory):
            os.mkdir(directory)

        if self.file_name is None:
            self.file_name = os.path.join(directory, "mnist")
            if self.train:
                self.file_name += "_train"
                (x, y), (_, _) = mnist.load_data()
            else:
                self.file_name += "_test"
                (_, _), (x, y) = mnist.load_data()
            if image:
                self.file_name += "_image"
            if label:
                self.file_name += "_label"
            self.file_name += str(self.slc) + ".tfrecord"

        if image:
            if label:
                if not os.path.exists(self.file_name):
                    save_data(x[self.slc], y[self.slc], self.file_name)
                return self._data_generator
            else:
                if not os.path.exists(self.file_name):
                    save_image(x[self.slc], self.file_name)
                return self._image_generator
        else:
            if not os.path.exists(self.file_name):
                save_label(y[self.slc], self.file_name)
            return self._label_generator


class VMnistDataset(MnistDataset):
    """
    Vertically splited mnist dataset
    """

    def __init__(self, slc, batch_size=32, train=True) -> None:
        super().__init__(batch_size, train)
        self.slc = slc
        self.row = self.slc[1].stop - self.slc[1].start

    def generator_builder(self, image=True, label=True):
        if not image and not label:
            return None

        directory = os.path.join(os.getcwd(), "data")
        if not os.path.exists(directory):
            os.mkdir(directory)

        if self.file_name is None:
            self.file_name = os.path.join(directory, "mnist")
            if self.train:
                self.file_name += "_train"
                (x, y), (_, _) = mnist.load_data()
            else:
                self.file_name += "_test"
                (_, _), (x, y) = mnist.load_data()
            if image:
                self.file_name += "_image"
            if label:
                self.file_name += "_label"
            self.file_name += str(self.slc) + ".tfrecord"

        if image:
            if label:
                if not os.path.exists(self.file_name):
                    save_data(x[self.slc], y, self.file_name)
                return self._data_generator
            else:
                if not os.path.exists(self.file_name):
                    save_image(x[self.slc], self.file_name)
                return self._image_generator
        else:
            if not os.path.exists(self.file_name):
                save_label(y, self.file_name)
            return self._label_generator
