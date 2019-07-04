"""Private prediction with a single clients"""
import tf_encrypted as tfe

from common import LogisticRegression, PredictionClient

num_features = 10

model = LogisticRegression(num_features)
prediction_client = PredictionClient('prediction-client', num_features)

x = prediction_client.provide_input()

y = model.forward(x)

reveal_output = prediction_client.receive_output(y)

with tfe.Session() as sess:
  sess.run(tfe.global_variables_initializer(), tag='init')

  sess.run(reveal_output, tag='predict')
