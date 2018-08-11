# TensorFlow Encrypted

This library provides a layer on top of TensorFlow for doing machine learning on encrypted data as initially described in [Secure Computations as Dataflow Programs](http://mortendahl.github.io/2018/03/01/secure-computation-as-dataflow-programs/). The bottom layer consists of basic protocols for computing with encrypted tensors, the next gives operations for basic machine learning tasks using these protocols, and the final layer exposes more packaged models for training and prediction on encrypted data.

## Usage

```python
import tensorflow as tf
import tensorflow_encrypted as tfe
```

# Examples

## Basic secure operations

This example shows how to do basic operations on encrypted tensors using the Pond protocol.

```python
import tensorflow_encrypted as tfe

# import the desired protocol for computing on encrypted data
from tensorflow_encrypted.protocol import Pond 

# this protocol requires two servers and a crypto producer;
# here we give their IP addresses
from tensorflow_encrypted.protocol import Server
server0 = Server('10.1.0.1')
server1 = Server('10.1.0.2')
crypto_producer = Server('10.1.0.3')

protocol = Pond(server0, server1, crypto_producer)

# 
a = prot.define_constant(np.array([4, 3, 2, 1]).reshape(2,2))
b = prot.define_constant(np.array([4, 3, 2, 1]).reshape(2,2))
c = a * b

d = prot.define_private_variable(np.array([1., 2., 3., 4.]).reshape(2,2))
e = prot.define_private_variable(np.array([1., 2., 3., 4.]).reshape(2,2))
# f = (d * .5 + e * .5)
f = d * e

from tensorflow_encrypted.config import session


with session(3) as sess:

    sess.run([d.initializer, e.initializer])

    print f.reveal().eval(sess)

    sess.run(prot.assign(d, f))
    sess.run(prot.assign(e, e))

    print f.reveal().eval(sess)

    g = prot.sigmoid(d)
    print g.reveal().eval(sess)
```

## Logistic regression

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

# Contributors

Several people have made significantly contributions to this project in one way or another:
- [Koen van der Veen](https://github.com/koenvanderveen)
- [Andrew Trask](https://github.com/iamtrask)