## Private Logistic Regression Tutorial

This tutorial shows how we can easily take a basic logistic regression problem and quickly begin making private predictions on the resulting model. We'll end up with a model that can run in both TF Encrypted and a [Trusted Execution Environment](https://en.wikipedia.org/wiki/Trusted_execution_environment) and has a reasonable runtime in both platforms!

### The Problem

The problem comes from a recent [Kaggle](https://www.kaggle.com/c/home-credit-default-risk) competition where data scientists were tasked with predicting whether a person was going to default on their house credit or not. We choose this problem because we knew we could get decent results with just a basic logistic regression model.

### The Data

The Kaggle competition provided 8 separate datasets. These datasets include detailed information about the loan, repayment history, credit cards, and previous credits provided by other financial institutions. We've gone ahead and done some feature engineering already and the final dataset can be downloaded from [here](https://storage.googleapis.com/tfe-examples-data/final_data.zip). We borrowed the script for feature engineering from [this](https://www.kaggle.com/ogrellier/lighgbm-with-selected-features) Kaggle kernel.

### The Model

To make a simple logistic regression model we just need one fully connected layer with no bias followed by a sigmoid activation function. We can make use of Tensorflow's Keras API to build this model. Here's what it looks like:

```python
model = keras.Sequential([
    layers.Dense(1, use_bias=False, activation='sigmoid', input_shape=[input_shape]),
])
```

During training we also need a loss function and an optimizer. For the loss function we use binary cross entropy because we're only choosing between two classes (i.e. defaulted: yes or no). We tried both Stochastic Gradient Descent and the Adam optimizers and found that Adam preformed best for this task. Both loss and the optimizer are added when compiling the Keras model, like such:

```python
model.compile(loss='binary_crossentropy',
              optimizer=tf.train.AdamOptimizer(),
              metrics=['accuracy'])
```

## Training the Model

Now we can train the model! Make sure you have cloned this repo, installed python 3.5 along with Tensorflow 1.12.0 and downloaded the final combined data downloaded from [here](https://storage.googleapis.com/tfe-examples-data/final_data.zip) and unzipped it into the directory containing this README.

Training the model can be done simply with:

```
$ python main.py
```

That could take a minute or two to finish. The longest part is loading all of the data into RAM.

Once finished you should see some output like so:

```
saved the frozen graph (ready for inference) at:  ./house_credit_default.pb
61500/61500 [==============================] - 0s 4us/step
Evaluation Loss: 0.24265269013730492
Accuracy: 0.9202276422802995
AUC:  0.76830494
```

There are some interesting metrics there that could probably better but the most important line for us is:

```
saved the frozen graph (ready for inference) at:  ./house_credit_default.pb
```

All this is saying is that the script saved the model and weights to a protobuf file which can then be used for prediction/inference. We'll make use of this later when doing private predictions in TF Encrypted and in a Trusted Execution Environment.

## Predicting in Plaintext

For now we'll just make use of the Keras API to make a prediction. We can do this by running the following:

```
$ python main.py --predict <row>
```

Where `<row>` is replaced by the row from the .csv file you'd like to make a prediction on! Like so:

```
$ python main.py --predict 10000
```

You should see an output like this:

```
Prediction: 0.04397624
```

Rounding down indicates that the client in question will not default.

## Predicting in TF Encrypted

Now we're ready to make a prediction inside TF Encrypted. First, we'll have to make sure TF Encrypted is installed. This can be done by following [these](../../docs/INSTALL.md) instructions.

The other thing we need to do is convert a row from the big .csv into a .npy file. This is currently the easiest way to do a quick prediction in tfe. There is a helper tool so we can just run:

```
$ python get_input.py --save_row <row>
```

Where `<row>` is replaced by the row from the .csv file you'd like to make a prediction on! Like so:

```
$ python get_input.py --save_row 10000
```

Once the input has been saved to the `input.npy` file we can make a prediction. There is a handy `run` script that makes running models locally easy. Run the following command to make the prediction:

```
$ ../../bin/run --model_name house_credit_default.pb --input_file input.npy
```

You should see output similar to the following:

```
initing!!!
running
[[0.41815186]]
```

Where 0.41815186 is the prediction.

## Predicting in Trusted Execution Environment

Next up is predicting inside of Trusted Execution Environment. To do this we'll have to clone [tf-trusted](https://github.com/dropoutlabs/tf-trusted) and follow the instructions in the README to build the necessary binaries. tf-trusted is an interface for running models inside of a TEE. tft uses TensorFlow [custom operations](https://www.tensorflow.org/guide/extend/op) to send [gRPC](https://grpc.io/) messages into the TEE device. tft uses [Asylo](https://asylo.dev/) to build the executable that runs inside of the TEE. Depending on the configuration of tft the TEE is either an Intel SGX Simulator or a real hardware Intel SGX device.

Running the model can be done similarly to how we ran the model in TF Encrypted but we need to make sure the gRPC server is running inside of the TEE. If not already done so open a new terminal and run the following command from with the tf-trusted repo:

```
$ docker run -it --rm \
  -v bazel-cache:/root/.cache/bazel \
  -v `pwd`:/opt/my-project \
  -w /opt/my-project \
  -p 50051:50051/tcp -p 50051:50051/udp \
  gcr.io/asylo-framework/asylo \
  bazel run --config=enc-sim //tf_trusted
```

Now, lets try running the model. First we need to make sure we install tft with pip.

```
$ pip install -e .
```

This makes sure that the `model_run.py` script is runnable from anywhere on your machine.

So from within this directory run:

```
$ model_run.py --model_file house_credit_default.pb \
               --input_file input.npy \
               --input_name "dense_input" \
               --output_name "dense/Sigmoid"
```

We've already generated the `.pb` and `.npy` files from the TF Encrypted example so they should already exists in this directory!

You should now see an output similar to this:

```
Prediction:  [[0.40583324]]
```

Where 0.40583324 is the prediction.
