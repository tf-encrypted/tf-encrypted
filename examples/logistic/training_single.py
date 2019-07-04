"""Private training on data from a single owner"""
import tf_encrypted as tfe

from common import DataOwner, ModelOwner, LogisticRegression

num_features = 10
training_set_size = 2000
test_set_size = 100
batch_size = 100
num_batches = (training_set_size // batch_size) * 10

model = LogisticRegression(num_features)
model_owner = ModelOwner('model-owner')
data_owner = DataOwner('data-owner',
                       num_features,
                       training_set_size,
                       test_set_size,
                       batch_size)

x_train, y_train = data_owner.provide_training_data()
x_test, y_test = data_owner.provide_testing_data()

reveal_weights_op = model_owner.receive_weights(model.weights)

with tfe.Session() as sess:
  sess.run([tfe.global_variables_initializer(),
            data_owner.initializer],
           tag='init')

  model.fit(sess, x_train, y_train, num_batches)
  model.evaluate(sess, x_test, y_test, data_owner)

  sess.run(reveal_weights_op, tag='reveal')
