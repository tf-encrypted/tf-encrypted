# Getting Started with TF Encrypted

This guide assumes that you have followed the installation instructions in [README](https://github.com/mortendahl/tf-encrypted).

## Contents

- [Introduction to the API](#introduction-to-the-api)
- [Private predictions](#private-predictions)
  - [Locally](#a-locally)
  - [With GCP](#b-gcp)
- [Private training](#private-training)

## Introduction to the API

TF Encrypted has a simple API to make it easy for data scientists to make private predictions and training.

To define a machine learning model, TF Encrypted and TensorFlow follow a very similar API.

<table>
<tr>
<th>TensorFlow</th>
<th>TF Encrypted</th>
</tr>
<tr>
<td><pre lang="python">layer0 = tf.matmul(x, w0) + b0
layer1 = tf.sigmoid(layer0)
layer2 = tf.matmul(layer, w1) + b1
prediction = layer2</pre>
</td>
<td><pre lang="python">layer0 = tfe.matmul(x, w0) + b0
layer1 = tfe.sigmoid(layer0)
layer2 = tfe.matmul(layer, w1) + b1
prediction = layer2</pre>
</td>
</tr>
</table>

To make private predictions and trainings, we have to define a ModelTrainer and a PredictionClient. The ModelTrainer is responsible for training the model, and then provides the encrypted weights. The PredictionClient will provide the private input that will be used to make a private prediction.


```python
model_trainer = ModelTrainer(config.get_player('model-trainer'))
prediction_client = PredictionClient(config.get_player('prediction-client'))
```

We can make a prediction as follows:
```python
# get model parameters as private tensors from model owner
w0, b0, w1, b1 = tfe.define_private_input(model_trainer)

# get prediction input from client
x = tfe.define_private_input(prediction_client)

# compute prediction
layer0 = tfe.matmul(x, w0) + b0
layer1 = tfe.sigmoid(layer0)
layer2 = tfe.matmul(layer, w1) + b1
prediction = layer2

# send prediction output back to client
prediction_op = tfe.define_output(prediction, prediction_client)
```

Two servers and a crypto producer are doing the actual computation on encrypted data, with only the client being able to decrypt the final result.




## Private Predictions
### a. Locally

With TF Encrypted, you can very easily make private predictions with a pre-trained model saved as a [protobuf](https://www.tensorflow.org/extend/tool_developers/) file.

1. In the [`examples/`](./examples/) folder run the following line of code in your terminal to train a convolution network on MNIST dataset, then save the model as a protobuf file :
```bash
python3 mnist_deep_cnn.py
```
If you prefer to skip this step, we have saved the trained model `mnist_model.pb` in [`models/test_data/`](./models/).

2. Make private prediction on an MNIST input by running the following code snippet:
```bash
../bin/run --protocol_name securenn --model_name mnist_model --batch_size 1 --input_file test_data/mnist_input.npy
```

You have just made a prediction without revealing anything about the input!

### b. GCP

**NOTE** It would be great to have an example as above with GCP where we import a protobuf file and run a prediction.

You can make private predictions with GCP as well. You can find a great example [here](https://github.com/mortendahl/tf-encrypted/tree/master/examples/mnist#remotely-on-gcp).

## Private Training

**NOTE** We currently don't have an example. We are planning to train/create a simple logistic regression as a demo..
