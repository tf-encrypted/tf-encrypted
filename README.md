# TensorFlow Encrypted

## Example

Training a logistic regression model using the two-party SPDZ protocol.

```python
import numpy as np
import tensorflow as tf
import tensorflow_encrypted as tfe

class MyInputProvider(tfe.NumpyInputProvider):

    def __init__(self, device_name, filename):
        self.filename = filename # TODO use
        super(MyInputProvider, self).__init__(device_name)

    @property
    def num_rows(self):
        return 1000

    @property
    def num_cols(self):
        return (2,1)

    def _build_data_generator(self):

        def generate_fake_training_data():
            np.random.seed(42)

            data_size = self.num_rows

            # generate features
            X0 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], data_size//2)
            X1 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], data_size//2)
            X = np.vstack((X0, X1)).astype(np.float32)

            # generate labels
            Y0 = np.zeros(data_size//2).reshape(-1, 1)
            Y1 = np.ones (data_size//2).reshape(-1, 1)
            Y = np.vstack((Y0, Y1)).astype(np.float32)

            # shuffle
            perm = np.random.permutation(len(X))
            X = X[perm]
            Y = Y[perm]
            
            return X, Y

        return generate_fake_training_data

input_providers = [
    MyInputProvider('/job:localhost/replica:0/task:0/device:CPU:3', './data_0/file.txt'),
    MyInputProvider('/job:localhost/replica:0/task:0/device:CPU:3', './data_1/file.txt'),
    MyInputProvider('/job:localhost/replica:0/task:0/device:CPU:3', './data_2/file.txt')
]

server0 = tfe.Server('/job:localhost/replica:0/task:0/device:CPU:0')
server1 = tfe.Server('/job:localhost/replica:0/task:0/device:CPU:1')
crypto_producer = tfe.CryptoProducer('/job:localhost/replica:0/task:0/device:CPU:2')

with tfe.protocol.TwoPartySPDZ(server0, server1, crypto_producer) as prot:

    logreg = tfe.estimator.LogisticClassifier(
        num_features=2
    )

    logreg.prepare_training_data(input_providers)
    logreg.train(epochs=1, batch_size=30)

    y_pred = logreg.predict(np.array([1., .5]))
    print tfe.recombine_and_decode(tfe.reconstruct(*y_pred))
```