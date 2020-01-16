import tensorflow as tf
import tf_encrypted as tfe


class LogisticRegression:
    def __init__(self, num_features, init_learning_rate=0.01):
        self.w = tfe.define_private_variable(
            tf.random_uniform([num_features, 1], -0.01, 0.01))
        self.b = tfe.define_private_variable(tf.zeros([1]))
        self.init_learning_rate = init_learning_rate

    @property
    def weights(self):
        return self.w, self.b

    def forward(self, x):
        with tf.name_scope("forward"):
            out = tfe.matmul(x, self.w) + self.b
            y_hat = tfe.sigmoid(out)
            return y_hat

    def backward(self, x, dy, learning_rate):
        batch_size = x.shape.as_list()[0]
        with tf.name_scope("backward"):
            dw = tfe.matmul(tfe.transpose(x), dy) / batch_size
            db = tfe.reduce_sum(dy, axis=0) / batch_size
            assign_ops = [
                tfe.assign(self.w, self.w - dw * learning_rate),
                tfe.assign(self.b, self.b - db * learning_rate)
            ]
            return assign_ops

    def loss_grad(self, y, y_hat):
        with tf.name_scope("loss-grad"):
            dy = y_hat - y
            return dy

    def fit_batch(self, x, y):
        with tf.name_scope("fit-batch"):
            y_hat = self.forward(x)
            dy = self.loss_grad(y, y_hat)
            fit_batch_op = self.backward(x, dy, self.init_learning_rate)
            return fit_batch_op

    def fit(self, sess, x, y, num_batches):
        fit_batch_op = self.fit_batch(x, y)
        for batch in range(num_batches):
            sess.run(fit_batch_op, tag="fit-batch")

    def loss(self, sess, x, y, player_name):
        def print_loss(y_hat, y):
            with tf.name_scope("print-loss"):
                loss = -y * tf.log(y_hat) - (1-y) * tf.log(1-y_hat)
                print_op = tf.print("Loss on {}:".format(player_name),
                                    loss)
                return print_op

        with tf.name_scope("loss"):
            y_hat = self.forward(x)
            print_loss_op = tfe.define_output(player_name,
                                              [y_hat, y],
                                              print_loss)
        sess.run(print_loss_op, tag="loss")

    def evaluate(self, sess, x, y, data_owner):
        """Return the accuracy"""
        def print_accuracy(y_hat, y) -> tf.Operation:
            with tf.name_scope("print-accuracy"):
                correct_prediction = tf.equal(tf.round(y_hat), y)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print_op = tf.print("Accuracy on {}:".format(data_owner.player_name),
                                    accuracy)
                return print_op

        with tf.name_scope("evaluate"):
            y_hat = self.forward(x)
            print_accuracy_op = tfe.define_output(data_owner.player_name,
                                                  [y_hat, y],
                                                  print_accuracy)

        sess.run(print_accuracy_op, tag='evaluate')


class FakeDataOwner:
    def __init__(self, player_name, num_features, train_set_size, test_set_size, batch_size):
        self.player_name = player_name
        self.num_features = num_features
        self.train_set_size = train_set_size
        self.test_set_size = test_set_size
        self.batch_size = batch_size
        self.train_initilizer = None
        self.test_initializer = None

    @property
    def initializer(self):
        return self.train_initilizer, self.test_initializer

    def provide_train_data_fake(self):
        x_raw = tf.random.uniform(minval=-0.5, maxval=0.5,
                                  shape=[self.train_set_size, self.num_features])
        # y_raw is created as a simple linear combination of x_raw's feature values
        y_raw = tf.cast(tf.reduce_mean(x_raw, axis=1, keepdims=True) > 0, dtype=tf.float32)

        train_set = tf.data.Dataset.from_tensor_slices((x_raw, y_raw))\
            .repeat()\
            .shuffle(buffer_size=self.batch_size)\
            .batch(self.batch_size)

        train_set_iterator = train_set.make_initializable_iterator()
        self.train_initilizer = train_set_iterator.initializer

        x, y = train_set_iterator.get_next()
        x = tf.reshape(x, [self.batch_size, self.num_features])
        y = tf.reshape(y, [self.batch_size, 1])

        return x, y

    def provide_train_features_fake(self):
        x_raw = tf.random.uniform(minval=-0.5, maxval=0.5,
                                  shape=[self.train_set_size, self.num_features])

        train_set = tf.data.Dataset.from_tensor_slices(x_raw) \
            .repeat() \
            .shuffle(buffer_size=self.batch_size) \
            .batch(self.batch_size)

        train_set_iterator = train_set.make_initializable_iterator()
        self.train_initilizer = train_set_iterator.initializer

        x = train_set_iterator.get_next()
        x = tf.reshape(x, [self.batch_size, self.num_features])

        return x

    def provide_train_targets_fake(self, *train_feature_sets):
        x = tf.concat(train_feature_sets, axis = 1)
        y = tf.cast(tf.reduce_mean(x, axis=1, keepdims=True) > 0, dtype=tf.float32)
        y = tf.reshape(y, [self.batch_size, 1])
        return y

    def provide_test_data_fake(self):
        x_raw = tf.random.uniform(
            minval=-.5,
            maxval=.5,
            shape=[self.test_set_size, self.num_features])

        y_raw = tf.cast(tf.reduce_mean(x_raw, axis=1) > 0, dtype=tf.float32)

        test_set = tf.data.Dataset.from_tensor_slices((x_raw, y_raw)) \
            .repeat() \
            .batch(self.test_set_size)

        test_set_iterator = test_set.make_initializable_iterator()
        self.test_initializer = test_set_iterator.initializer

        x, y = test_set_iterator.get_next()
        x = tf.reshape(x, [self.test_set_size, self.num_features])
        y = tf.reshape(y, [self.test_set_size, 1])

        return x, y

    def provide_test_features_fake(self):
        x_raw = tf.random.uniform(
            minval=-.5,
            maxval=.5,
            shape=[self.test_set_size, self.num_features])

        test_set = tf.data.Dataset.from_tensor_slices(x_raw) \
            .repeat() \
            .batch(self.test_set_size)

        test_set_iterator = test_set.make_initializable_iterator()
        self.test_initializer = test_set_iterator.initializer

        x = test_set_iterator.get_next()
        x = tf.reshape(x, [self.test_set_size, self.num_features])

        return x

    def provide_test_targets_fake(self, *test_feature_sets):
        x = tf.concat(test_feature_sets, axis = 1)
        y = tf.cast(tf.reduce_mean(x, axis=1, keepdims=True) > 0, dtype=tf.float32)
        y = tf.reshape(y, [self.test_set_size, 1])
        return y


class DataSchema:
    def __init__(self, field_types, field_defaults):
        self.field_types = field_types
        self.field_defaults = field_defaults

    @property
    def field_num(self):
        return len(self.field_types)


class DataOwner:
    def __init__(self, player_name, local_data_file, data_schema,
                 header=False, index=False, field_delim=',', na_values=['nan'], batch_size=128):
        self.player_name = player_name
        self.local_data_file = local_data_file
        self.data_schema = data_schema
        self.batch_size = batch_size
        self.header = header
        self.index = index
        self.na_values = na_values
        self.field_delim = field_delim

    def provide_data_experimental(self):
        """
        Use TF's CsvDataset to load local data, but it is too slow.
        Please use `self.provide_data` instead.
        """
        dataset = tf.data.experimental.CsvDataset(
            self.local_data_file,
            [tf.constant([self.data_schema.field_defaults[i]], dtype=self.data_schema.field_types[i])
             for i in range(self.data_schema.field_num)],
            header=self.header,
            field_delim=self.field_delim,
            select_cols=list(range(self.data_schema.field_num)) if not self.index else list(range(1, self.data_schema.field_num+1))
        )  \
            .repeat() \
            .batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        batch = iterator.get_next()
        batch = tf.transpose(tf.stack(batch, axis=0))
        return batch

    def provide_data(self):

        def decode(line):
            fields = tf.string_split([line], self.field_delim).values
            if self.index: # Skip index
                fields = fields[1:]
            fields = tf.regex_replace(fields, '|'.join(self.na_values), 'nan')
            fields = tf.string_to_number(fields, tf.float32)
            return fields

        def fill_na(fields, fill_values):
            fields = tf.where(tf.is_nan(fields), fill_values, fields)
            return fields

        dataset = tf.data.TextLineDataset(self.local_data_file)
        if self.header: # Skip header
            dataset = dataset.skip(1)
        dataset = dataset\
            .map(decode)\
            .map(lambda x: fill_na(x, self.data_schema.field_defaults))\
            .repeat()\
            .batch(self.batch_size)

        iterator = dataset.make_one_shot_iterator()
        batch = iterator.get_next()
        batch = tf.reshape(batch, [self.batch_size, self.data_schema.field_num])
        return batch


class ModelOwner:
    def __init__(self, player_name):
        self.player_name = player_name

    def receive_weights(self, *weights):
        return tf.print("Weights on {}:\n".format(self.player_name), weights)


class PredictionClient:
    def __init__(self, player_name, num_features):
        self.player_name = player_name
        self.num_features = num_features

    def provide_input_fake(self):
        return tf.random.uniform(minval=-0.5, maxval=0.5, dtype=tf.float32,
                                 shape=[1, self.num_features])

    def receive_output(self, result):
        return tf.print("Result on {}:".format(self.player_name), result)


class LossDebugger:
    def __init__(self, player_name):
        self.player_name = player_name

    def loss(self, model, x, y):
        def print_loss(y_hat, y):
            with tf.name_scope("print-loss"):
                loss = -y * tf.log(y_hat) - (1-y) * tf.log(1-y_hat)
                loss = tf.reduce_mean(loss)
                print_op = tf.print("Loss on {}:".format(self.player_name),
                                    loss)
                return print_op

        with tf.name_scope("loss"):
            y_hat = model.forward(x)
            print_loss_op = tfe.define_output(self.player_name,
                                              [y_hat, y],
                                              print_loss)
        return print_loss_op
