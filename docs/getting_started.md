# TensorFlow Encrypted

This guide assumes that you have followed the installation instructions in [README](https://github.com/mortendahl/tf-encrypted).

- [Introduction to Tensorflow Encrypted API](#introduction-to-tensorflow-encrypted-api)
- [Making private predictions](#make-private-prediction)
  - [Locally](#locally)
  - [With GCP](#with-gcp)
- [Making private trainings](#make-private-training)

## Introduction to TensorFlow Encrypted API

tf-encrypted has a simple api to make it easy for data scientists to make private predictions and trainings.


To define a machine learning model, tf-encrypted and TensorFlow follow a very similar API.

TensorFlow             |  TensorFlow Encrypted
:-------------------------:|:-------------------------:
<img src="images/tf_layers.png" width="400" height="120" />  |  <img src="images/tfe_layers.png" width="400" height="120" />

To make private predictions and trainings, we have to define a ModelTrainer and a PredictionClient. The ModelTrainer is responsible for training the model, and then provides the encrypted weights. The PredictionClient will provide the private input that will be used to make a private prediction.

<img src="images/modelTrainer_PredictionClient.png" width="700" height="70" />

We indicate tf-encrypted to perform secure computation with the Pond protocol as follows:
<img src="./images/pond.png" width="600" height="23" />

Finally we can make a prediction as follows:
<img src="./images/model.png" width="720" height="400" />

Two servers and a crypto producer are doing the actual computation on encrypted data, with only the client being able to decrypt the final result.




## Making Private Predictions
### a. Locally

With Tensorflow Encrypted, you can very easily make private predictions with a pre-trained model saved as a [protobuf](https://www.tensorflow.org/extend/tool_developers/) file.

1. In the [`examples/`](./examples/) folder run the following line of code in your terminal to train a convolution network on MNIST dataset, then save the model as a protobuf file :
```bash
python3 mnist_deep_cnn.py
```
If you prefer to skip this step, we have saved the trained model `mnist_model.pb` in [`examples/test_data/`](./examples/test_data/).

2. Make private prediction on an MNIST input by running the following code snippet:
```bash
../bin/run test_data/mnist_model.pb test_data/mnist_input.npy
```

You have just made a prediction without revealing anything about the input!

### b. GCP

**NOTE** It would be great to have an example as above with GCP where we import a protobuf file and run a prediction.

You can make private predictions with GCP as well. You can find a great example [here](https://github.com/mortendahl/tf-encrypted/tree/master/examples/mnist#remotely-on-gcp).

## Making Private Trainings

**NOTE** We currently don't have an example. We are planning to train/create a simple logistic regression as a demo..
