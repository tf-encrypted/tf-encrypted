import tf_encrypted as tfe

from common import LogisticRegression, PredictionClient

num_features = 10

model = LogisticRegression(num_features)
prediction_client = PredictionClient('prediction-client', num_features)

x = tfe.define_private_input(prediction_client.player_name, prediction_client.provide_input)

y = model.forward(x)

reveal_output = tfe.define_output(prediction_client.player_name, y, prediction_client.receive_output)

with tfe.Session() as sess:
  sess.run(tfe.global_variables_initializer(), tag='init')

  sess.run(reveal_output, tag='predict')
