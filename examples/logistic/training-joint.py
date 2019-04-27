import tf_encrypted as tfe

from common import DataOwner, ModelOwner, LogisticRegression

num_features = 10
training_set_size = 2000
test_set_size = 100
batch_size = 100
num_batches = (training_set_size // batch_size) * 10

model_owner = ModelOwner('model-owner')
data_owner_0 = DataOwner('data-owner-0', num_features, training_set_size, test_set_size, batch_size // 2)
data_owner_1 = DataOwner('data-owner-1', num_features, training_set_size, test_set_size, batch_size // 2)

tfe.set_protocol(tfe.protocol.Pond(
    tfe.get_config().get_player(data_owner_0.player_name),
    tfe.get_config().get_player(data_owner_1.player_name)
))

x_train_0, y_train_0 = tfe.define_private_input(data_owner_0.player_name, data_owner_0.provide_training_data)
x_train_1, y_train_1 = tfe.define_private_input(data_owner_1.player_name, data_owner_1.provide_training_data)

x_test_0, y_test_0 = tfe.define_private_input(data_owner_0.player_name, data_owner_0.provide_testing_data)
x_test_1, y_test_1 = tfe.define_private_input(data_owner_1.player_name, data_owner_1.provide_testing_data)

x_train = tfe.concat([x_train_0, x_train_1], axis=0)
y_train = tfe.concat([y_train_0, y_train_1], axis=0)

model = LogisticRegression(num_features)
reveal_weights_op = tfe.define_output(model_owner.player_name, model.weights, model_owner.receive_weights)

with tfe.Session() as sess:
    sess.run([tfe.global_variables_initializer(), data_owner_0.initializer, data_owner_1.initializer], tag='init')

    model.fit(sess, x_train, y_train, num_batches)
    # TODO(Morten)
    # each evaluation results in nodes for a forward pass being added to the graph;
    # maybe there's some way to avoid this, even if it means only if the shapes match
    model.evaluate(sess, x_test_0, y_test_0, data_owner_0)
    model.evaluate(sess, x_test_1, y_test_1, data_owner_1)

    sess.run(reveal_weights_op, tag='reveal')
