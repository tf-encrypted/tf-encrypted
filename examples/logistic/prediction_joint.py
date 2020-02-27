"""Private prediction on combined features from several clients"""
import tf_encrypted as tfe
from common import LogisticRegression
from common import PredictionClient

num_features = 10

model = LogisticRegression(num_features)
prediction_client_0 = PredictionClient("prediction-client-0", num_features // 2)
prediction_client_1 = PredictionClient("prediction-client-1", num_features // 2)
result_receiver = prediction_client_0

x_0 = prediction_client_0.provide_input()
x_1 = prediction_client_1.provide_input()
x = tfe.concat([x_0, x_1], axis=1)

y = model.forward(x)


reveal_output = result_receiver.receive_output(y)

with tfe.Session() as sess:
    sess.run(tfe.global_variables_initializer(), tag="init")

    sess.run(reveal_output, tag="predict")
