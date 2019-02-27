import sys

import tensorflow as tf
import tf_encrypted as tfe

from convert import decode

if len(sys.argv) > 1:
    # config file was specified
    config_file = sys.argv[1]
    config = tfe.RemoteConfig.load(config_file)
    tfe.set_config(config)
    tfe.set_protocol(tfe.protocol.Pond())

session_target = sys.argv[2] if len(sys.argv) > 2 else None


class ModelOwner:

    LEARNING_RATE = 0.1
    ITERATIONS = 60000 // 30

    def __init__(self, player_name):
        self.player_name = player_name

        with tf.device(tfe.get_config().get_player(player_name).device_name):
            self._initialize_weights()

    def _initialize_weights(self):
        with tf.name_scope('parameters'):
            self.w0 = tf.Variable(tf.random_normal([28 * 28, 512]))
            self.b0 = tf.Variable(tf.zeros([512]))
            self.w1 = tf.Variable(tf.random_normal([512, 10]))
            self.b1 = tf.Variable(tf.zeros([10]))

    def _build_model(self, x, y):
        w0 = self.w0.read_value()
        b0 = self.b0.read_value()
        w1 = self.w1.read_value()
        b1 = self.b1.read_value()
        params = (w0, b0, w1, b1)

        layer0 = tf.matmul(x, w0) + b0
        layer1 = tf.nn.sigmoid(layer0)
        layer2 = tf.matmul(layer1, w1) + b1
        predictions = layer2

        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=predictions, labels=y))
        grads = tf.gradients(ys=loss, xs=params)
        return predictions, loss, grads

    def build_training_model(self, x, y):
        """
        This method will be called once by all data owners
        to create a local gradient computation on their machine.
        """
        _, _, grads = self._build_model(x, y)
        return grads

    def _build_validation_model(self, x, y):
        predictions, loss, _ = self._build_model(x, y)
        most_likely = tf.argmax(predictions, axis=1)
        return most_likely, loss

    def _build_data_pipeline(self):

        def normalize(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            return image, label

        dataset = tf.data.TFRecordDataset(["./data/train.tfrecord"])
        dataset = dataset.map(decode)
        dataset = dataset.map(normalize)
        dataset = dataset.batch(50)
        dataset = dataset.take(1)  # keep validating on the same items
        dataset = dataset.repeat()

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def update_model(self, *grads):
        params = [self.w0, self.b0, self.w1, self.b1]
        grads = [tf.cast(grad, tf.float32) for grad in grads]
        with tf.name_scope('update'):
            update_op = tf.group(*[
                param.assign(param - grad * self.LEARNING_RATE)
                for param, grad in zip(params, grads)
            ])
            # return update_op

        with tf.name_scope('validate'):
            x, y = self._build_data_pipeline()
            y_hat, loss = self._build_validation_model(x, y)

            with tf.control_dependencies([update_op]):
                return tf.print('expect', loss, y, y_hat, summarize=50)


class DataOwner:

    BATCH_SIZE = 30

    def __init__(self, player_name, build_training_model):
        self.player_name = player_name
        self._build_training_model = build_training_model

    def _build_data_pipeline(self):

        def normalize(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            return image, label

        dataset = tf.data.TFRecordDataset(["./data/train.tfrecord"])
        dataset = dataset.map(decode)
        dataset = dataset.map(normalize)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.BATCH_SIZE)

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def compute_gradient(self):

        with tf.name_scope('data_loading'):
            x, y = self._build_data_pipeline()

        with tf.name_scope('gradient_computation'):
            grads = self._build_training_model(x, y)

        return grads


model_owner = ModelOwner('model-owner')
data_owners = [
    DataOwner('data-owner-0', model_owner.build_training_model),
    DataOwner('data-owner-1', model_owner.build_training_model),
    DataOwner('data-owner-2', model_owner.build_training_model),
]

model_grads = zip(*(
    tfe.define_private_input(data_owner.player_name, data_owner.compute_gradient)
    for data_owner in data_owners
))

with tf.name_scope('secure_aggregation'):
    aggregated_model_grads = [
        tfe.add_n(grads) / len(grads)
        for grads in model_grads
    ]

iteration_op = tfe.define_output(model_owner.player_name, aggregated_model_grads, model_owner.update_model)

with tfe.Session(target=session_target) as sess:
    sess.run(tf.global_variables_initializer(), tag='init')

    for i in range(model_owner.ITERATIONS):
        if i % 100 == 0:
            print("Iteration {}".format(i))
            sess.run(iteration_op, tag='iteration')
        else:
            sess.run(iteration_op)
