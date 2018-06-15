# TensorFlow Encrypted

## Usage

```python
import tensorflow as tf
import tensorflow_encrypted as tfe
```

## Example

Training a logistic regression model using the two-server SPDZ protocol.

We first define how our input providers are going to preprocess and prepare the training data; in this example we simply let each one generate a fake dataset. Note that `_build_data_generator` returns a generator function that is later executed by TensorFlow on the specified device as part of graph execution.

```python
import numpy as np

class MyInputProvider(tfe.NumpyInputProvider):

    def __init__(self, device_name):
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
```

We create five instances of these running on different hosts:

```python
input_providers = [
    MyInputProvider('10.0.0.1'),
    MyInputProvider('10.0.0.2'),
    MyInputProvider('10.0.0.3'),
    MyInputProvider('10.0.0.4'),
    MyInputProvider('10.0.0.5')
]
```

Next we set up two servers that will jointly do the encrypted training and prediction, leaving the input providers to only be needed initially to distribute their encrypted data:

```python
server1 = tfe.Server('10.1.0.1')
server2 = tfe.Server('10.1.0.2')
```

Finally, as part of the joint computation we also need certain cryptographic raw material that can processed in advance; for this example we let a single host produce this:

```python
crypto_producer = tfe.CryptoProducer('10.2.0.1')
```

Finally, we instantiate the protocol in order to:
1) transfer the secret shared training set from the input providers to the two servers
2) train a logistic model shared between the two servers
3) perform an sample prediction

```python
with tfe.protocol.TwoServerSPDZ(server1, server2, crypto_producer):

    with tfe.session() as sess:

        logreg = tfe.estimator.LogisticClassifier(
            session=sess,
            num_features=2,
        )

        logreg.prepare_training_data(input_providers)
        
        logreg.train(epochs=10, batch_size=30)

        y = logreg.predict(x)
        print y.reveal()
```