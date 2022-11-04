import tensorflow as tf


class LogisticArtificialDataset:
    """
    Artificially generated dataset for logistic regression
    """

    def __init__(self, batch_size=32, train=True) -> None:
        self.train = train
        self.batch_size = batch_size
        if train:
            self.num_samples = 2000
        else:
            self.num_samples = 100
        self.num_features = 10
        self.num_classes = 2
        self.iterations = self.num_samples // self.batch_size
        self.file_name = None

    def reset_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.iterations = self.num_samples // self.batch_size

    @property
    def batch_shape(self):
        return [self.batch_size, self.num_features]

    def _data_generator(self):
        def norm(x, y):
            return tf.cast(x, tf.float32), tf.expand_dims(y, 0)

        x_raw = tf.random.uniform(
            minval=-0.5, maxval=0.5, shape=[self.num_samples, self.num_features]
        )

        y_raw = tf.cast(tf.reduce_mean(x_raw, axis=1) > 0, dtype=tf.float32)

        dataset = tf.data.Dataset.from_tensor_slices((x_raw, y_raw))
        dataset = (
            dataset.map(norm)
            .cache()
            .shuffle(buffer_size=self.batch_size)
            .batch(self.batch_size, drop_remainder=True)
        )  # drop remainder because we need to fix batch size in private model
        if self.train:
            dataset = dataset.repeat()
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        data_iter = iter(dataset)

        return data_iter

    def generator_builder(self):
        return self._data_generator
