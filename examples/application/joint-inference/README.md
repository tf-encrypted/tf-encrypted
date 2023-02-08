# Private Joint Inference

This example illustrates how TF Encrypted can be used to perform private joint inference with various networks and datasets. It also shows how to integrate with ordinary TensorFlow, seamlessly linking local computations with secure computations.

Our scenario is split into two phases.

In the first phase, a *model owner* trains a model locally and sends encryptions of the resulting weights to three *compute servers* as used by the default ABY3 secure computation protocol. The training is done using an ordinary TensorFlow computation and the encrypted weights are cached on the servers for repeated use.

In the second phase, a *prediction client* sends an encryption of its input to the servers, who perform a secure computation over the weights and input to arrive at an encrypted prediction, which is finally sent back to the client and decrypted. The client also uses ordinary TensorFlow computations to apply pre- and post-processing.

<p align="center"><img src="./flow.png" style="width: 50%;"/></p>

The goal of the example is to show that the computations done by the servers can be performed entirely on encrypted data, at no point being able to decrypt any of the values. For this reason we can see the weights as a private input from the model owner and the prediction input as a private input from the prediction client.

## Computation

`ModelOwner` trains a model locally with tensorflow API and share model weights privately to three *compute servers*.

```python
@tfe.local_computation(player_name="model-owner", name_scope="share_model_weights")
def share_model_weights(model_name, data_name):
    # model owner train a model and share its weights
    Dataset = globals()[data_name.capitalize() + "Dataset"]
    train_dataset = Dataset(batch_size=128)
    data_iter = train_dataset.generator_builder()()
    model = globals()[model_name](train_dataset.batch_shape, train_dataset.num_classes, private=False)
    if train_dataset.num_classes > 1:
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        metrics=["categorical_accuracy"]
    else:
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics=["binary_accuracy"]
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    # optimizer = tf.keras.optimizers.AMSgrad(learning_rate=0.001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer, loss, metrics=metrics)
    for i in range(train_dataset.iterations):
        data = next(data_iter)
        model.train_step(data)
    return model.weights
```

`PredictionClient` likewise share its prediction input privately to three *compute servers* by define a DataOwner. but also a post-processing computation that applies an softmax on the decrypted result before printing to the screen.

```python
Dataset = globals()[args.data_name.capitalize() + "Dataset"]
# set prediction client
test_dataset = Dataset(batch_size=100, train=False)
prediction_client = DataOwner(
    config.get_player("prediction-client"), 
    test_dataset.generator_builder(label=False)
)

@tfe.local_computation(player_name="prediction-client", name_scope="receive_prediction")
def receive_prediction(prediction):
    # simply print prediction result
    prediction = tf.nn.softmax(prediction)
    tf.print("Prediction result:", prediction)
```

Instances of these are then linked together in a secure computation performing a prediction, treating both the weights and the prediction input as private values.

```python
    # share model weihgts
    model_weights = share_model_weights(args.model_name, args.data_name)

    print("Set model weights")
    model = globals()[args.model_name](test_dataset.batch_shape, test_dataset.num_classes)
    model.set_weights(model_weights)

    print("perform predict")
    result = model.predict(x=prediction_client.provide_data(), reveal=False)
    receive_prediction(result)
```

## Running

Start the private joint inference with the startup script `run-remote.sh`, and you must explicitly specify which model to inference and which dataset to use.

```sh
./examples/application/joint-inference/run-remote.sh network_a Mnist
```

You can also specify which tfe protocol to use and which remote config file to use

```sh
./examples/application/joint-inference/run-remote.sh network_a Mnist --protocol ABY3 --config config.json
```

By default, inference uses 64 bits for secret sharing, this gives enough precision in most cases.
We also give a option to use 128 bits for secret sharing by setting `--precision high`,
this will give you more precision, but at a cost of more computation time.

```sh
./examples/benchmark/aby3_profile/run-remote.sh network_a Mnist --precision high
```

See more details in the [documentation](/docs/RUNNING.md).
