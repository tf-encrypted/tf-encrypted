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
        """Prepare input data for prediction."""
        prediction_input, expected_result = self._build_data_pipeline().get_next()
        prediction_input = tf.reshape(
            prediction_input, shape=(self.BATCH_SIZE, ModelOwner.FLATTENED_DIM))
        return prediction_input

    def receive_output(self, logits: tf.Tensor) -> tf.Operation:
        prediction = tf.argmax(logits, axis=1)
        op = tf.print("Result", prediction, summarize=self.BATCH_SIZE)
        return op
```

Instances of these are then linked together in a secure computation performing a prediction, treating both the weights and the prediction input as private values.

```python
  model_owner = ModelOwner(player_name="model-owner")
  prediction_client = PredictionClient(player_name="prediction-client")

  with tfe.protocol.SecureNN():
    batch_size = PredictionClient.BATCH_SIZE
    flat_dim = ModelOwner.IMG_ROWS * ModelOwner.IMG_COLS

    model = tfe.keras.Sequential()
    model.add(tfe.keras.layers.Dense(512, batch_input_shape=batch_input_shape))
    model.add(tfe.keras.layers.Activation('relu'))
    model.add(tfe.keras.layers.Dense(10))
    model.set_weights(params)

    # get prediction input from client
    x = tfe.define_private_input(prediction_client.player_name,
                                 prediction_client.provide_input)
    logits = model(x)

  # send prediction output back to client
  prediction_op = tfe.define_output(prediction_client.player_name,
                                    logits,
                                    prediction_client.receive_output)
```

Finally, the computation is executed using a `tfe.Session` following the typical TensorFlow pattern

```python
with tfe.Session() as sess:
    sess.run(prediction_op)
```

## Running

Make sure to have the training and test data sets downloaded before running the example:

```sh
python3 examples/mnist/download.py
```

which will place the converted files in the `./data` subdirectory.

To then run locally use:

```sh
python3 examples/mnist/run.py
```

or remotely using:

```sh
python3 examples/mnist/run.py config.json
```

See more details in the [documentation](/docs/RUNNING.md).
