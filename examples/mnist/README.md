# Private Predictions for MNIST

This example illustrates how TF Encrypted can be used to perform private predictions using a simple neural network on the MNIST data set. It also shows how to integrate with ordinary TensorFlow, seamlessly linking local computations with secure computations.

Our scenario is split into two phases.

In the first phase, a *model owner* trains a model locally and sends encryptions of the resulting weights to three *compute servers* as used by the default Pond secure computation protocol. The training is done using an ordinary TensorFlow computation and the encrypted weights are cached on the servers for repeated use.

In the second phase, a *prediction client* sends an encryption of its input to the servers, who perform a secure computation over the weights and input to arrive at an encrypted prediction, which is finally sent back to the client and decrypted. The client also uses ordinary TensorFlow computations to apply pre- and post-processing.

<p align="center"><img src="./flow.png" style="width: 50%;"/></p>

The goal of the example is to show that the computations done by the servers can be performed entirely on encrypted data, at no point being able to decrypt any of the values. For this reason we can see the weights as a private input from the model owner and the prediction input as a private input from the prediction client.

## Computation

The code is structured around a `ModelOwner` and `PredictionClient` class.

`ModelOwner` builds a data pipeline for training data using the TensorFlow Dataset API and performs training using TensorFlow's built in components.

```python
class ModelOwner:

    def provide_input(self) -> tf.Tensor:
        # training
        training_data = self._build_data_pipeline()
        weights = self._build_training_graph(training_data)
        return weights
```

`PredictionClient` likewise builds a data pipeline for prediction inputs but also a post-processing computation that applies an argmax on the decrypted result before printing to the screen.

```python
class PredictionClient:

    def provide_input(self) -> tf.Tensor:
        prediction_input = self._build_data_pipeline().get_next()
        # pre-processing
        prediction_input = tf.reshape(prediction_input, shape=(self.BATCH_SIZE, 28 * 28))
        return prediction_input

    def receive_output(self, likelihoods: tf.Tensor) -> tf.Operation:
        # post-processing
        prediction = tf.argmax(likelihoods, axis=1)
        op = tf.print("Result", prediction, summarize=self.BATCH_SIZE)
        return op
```

Instances of these are then linked together in a secure computation performing a prediction, treating both the weights and the prediction input as private values.

```python
model_owner = ModelOwner('model-owner')
prediction_client = PredictionClient('prediction-client')

w0, b0, w1, b1 = tfe.define_private_input(model_owner.player_name, model_owner.provide_input)
x = tfe.define_private_input(prediction_client.player_name, prediction_client.provide_input)

layer0 = tfe.matmul(x, w0) + b0
layer1 = tfe.sigmoid(layer0)
logits = tfe.matmul(layer1, w1) + b1

prediction_op = tfe.define_output(prediction_client.player_name, logits, prediction_client.receive_output)
```

Finally, the computation is executed using a `tfe.Session` following the typical TensorFlow pattern

```python
with tfe.Session() as sess:
    sess.run(prediction_op)
```

## Running

This is the easiest way of running the example, however it doesn't provide any security and doesn't give accurate performance numbers (may even be slower than [running on GCP](#remotely-on-gcp)).

The first step is to download and convert the MNIST dataset: from the project root directory simply run
```shell
python3 ./examples/mnist/download.py
```
which will place the converted files in the `./data` subdirectory. Next, to execute the example run
```shell
python3 ./examples/mnist/run.py
```
again from the project root directory.

To get debugging and profiling data on the run use
```shell
TFE_STATS=1 python3 ./examples/mnist/run.py
```
instead, which will write files to `/tmp/tensorboard` that can be inspected using TensorBoard, e.g.
```shell
tensorboard --logdir=/tmp/tensorboard
```
and navigating to http://localhost:6006 in a browser.
