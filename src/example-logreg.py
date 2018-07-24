import numpy as np
import tensorflow as tf
import time
import tensorflow_encrypted as tfe
from tensorflow_encrypted.inputs import FakeInputProvider

input_providers = [
    FakeInputProvider('/job:localhost/replica:0/task:0/device:CPU:3'),
    FakeInputProvider('/job:localhost/replica:0/task:0/device:CPU:4'),
    FakeInputProvider('/job:localhost/replica:0/task:0/device:CPU:5')
]

server0 = tfe.Server('/job:localhost/replica:0/task:0/device:CPU:0')
server1 = tfe.Server('/job:localhost/replica:0/task:0/device:CPU:1')
crypto_producer = tfe.CryptoProducer('/job:localhost/replica:0/task:0/device:CPU:2')

with tfe.protocol.TwoPartySPDZ(server0, server1, crypto_producer) as prot:

    with tfe.session(num_players=6) as sess:
        begin = time.time()

        print("Creating a classifier...")
        logreg = tfe.estimator.LogisticClassifier(
            session=sess,
            num_features=2
        )

        print("Preparing training data...")
        logreg.prepare_training_data(input_providers)
        
        print("Training...")
        logreg.train(epochs=100, batch_size=30)

        print time.time() - begin

        # print logreg.predict(np.array([1., .5]))
