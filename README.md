# TensorFlow Encrypted

This library provides a layer on top of TensorFlow for doing machine learning on encrypted data as initially described in [Secure Computations as Dataflow Programs](https://mortendahl.github.io/2018/03/01/secure-computation-as-dataflow-programs/) and is structured into roughly three levels: 

- basic operations for computing on encrypted tensors
- basic machine learning components using these
- ready-to-use models for private machine learning

with the aim of making it easy for researchers and practitioners to experiment in a familiar framework and without being an expert in both machine learning and cryptography.

## Usage

`tf-encrypted` can be imported next to TensorFlow, giving access to extra operations for constructing regular TensorFlow graphs that operate on encrypted data. Since secure computation protocols by nature involve more than one party, each protocol instance takes a list of hostnames as input (three for the `Pond` protocol) and then exposes methods for creating private variables etc. along the lines of traditional TensorFlow programs. Simple wrappers around `tf.Session` can finally be used to execute these.

```python
import tensorflow as tf
import tensorflow_encrypted as tfe

servers = [
    tfe.protocol.Player('10.0.0.1'),
    tfe.protocol.Player('10.0.0.2'),
    tfe.protocol.Player('10.0.0.3')
]

with tfe.protocol.Pond(*servers) as prot:

    w = prot.private_variable(weight_values)
    b = prot.private_variable(bias_values)

    x = prot.private_placeholder(prediction_shape)
    y = prot.sigmoid(prot.dot(x, w) + b)

    res = y.reveal()

with tfe.remote_session() as sess:

    print(res.eval(sess))
```

Note that higher level estimators are also available, such as `tfe.estimator.LogisticClassifier`.

# Examples

<em>(more coming)</em>

## Logistic regression

```python
import tensorflow as tf
import tensorflow_encrypted as tfe

servers = [
    tfe.protocol.Player('10.0.0.1'),
    tfe.protocol.Player('10.0.0.2'),
    tfe.protocol.Player('10.0.0.3')
]

training_providers = [
    tfe.protocol.InputProvider('10.1.1.1'),
    tfe.protocol.InputProvider('10.1.1.2'),
    tfe.protocol.InputProvider('10.1.1.3'),
    tfe.protocol.InputProvider('10.1.1.4'),
    tfe.protocol.InputProvider('10.1.1.5')
]

prediction_providers = [
    tfe.protocol.InputProvider('10.2.2.1'),
    tfe.protocol.InputProvider('10.2.2.2')
]

with tfe.remote_session() as sess:

    with tfe.protocol.Pond(*servers) as prot:

        # instantiate estimator
        logreg = tfe.estimator.LogisticClassifier(
            session=sess,
            num_features=2
        )

        # perform private training based on data from providers
        logreg.prepare_training_data(training_providers)
        logreg.train(epochs=100, batch_size=30)

        # perform private prediction
        prediction = logreg.predict(prediction_providers)
```

# Contributions

Several people have made contributions to this project in one way or another (in alphabetical order):
- [Andrew Trask](https://github.com/iamtrask)
- [Koen van der Veen](https://github.com/koenvanderveen)
