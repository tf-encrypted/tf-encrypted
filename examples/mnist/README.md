
# Overview

This example not only illustrates how MNIST digit predictions can be done on encrypted data, but also how TensorFlow Encrypted can be integrated with ordinary TensorFlow.

In particular, we here have a separate *model trainer* that provides encrypted weights to our two compute servers, all running on different hosts. And while these weights might typically simply be read from disk, in this example the model trainer runs an ordinary TensorFlow computation locally to first train these weigths when requested to by the servers, who then catches them for future predictions. Likewise, our *prediction client* sends the input and receives the prediction directly in TensorFlow, allowing it to locally both pre-process and post-process using ordinary TensorFlow mechanisms.

Concretely, a `ModelTrainer` and `PredictionClient` class represent the two parties above, both extending `tfe.io.InputProvider` and the latter also `tfe.io.OutputReceiver`. When private values are needed from either their `provide_input()` methods are executed *locally* on their associated host, and the resulting `tf.Tensors` are encrypted before sending them to the servers. This input and output behaviour is then connected with the secure computation simply by
```python
w0, b0, w1, b1 = prot.define_private_input(model_trainer)
x, = prot.define_private_input(prediction_client)
```
and
```python
prediction_op = prot.define_output([prediction], prediction_client)
```
where `prediction` is the encrypted result of the prediction on `x`.

# Running

We here give two ways of running this example.

## Locally for testing

This is the easiest way of running the example, however it doesn't provide any security and doesn't give accurate performance numbers (may even be slower than [running on GCP](#remotely-on-gcp)).

The first step is to download and convert the MNIST dataset: from the project root directory simply run
```shell
python3 ./examples/mnist/download.py
```
which will place the converted files in the `./data` subdirectory. Next, to execute the example run
```shell
python3 ./examples/mnist/run.py
```
again from the project root directory. This will also put debugging and profiling data in `/tmp/tensorboard` which can be inspected using TensorBoard, e.g.
```shell
tensorboard --logdir=/tmp/tensorboard
```
and navigating to http://localhost:6006 in a browser.

## Remotely on GCP

This way of running the example is slightly more involved but will give actual performance numbers (and security). All steps here assume that the [Cloud SDK](https://cloud.google.com/sdk/) has already been installed (for macOS this may be done via e.g. Homebrew: `brew cask install google-cloud-sdk`) and that a [tfe-image](../../tools/gcp/) has already been created.

### Setup instances

This example needs the following instances: `master`, `server0`, `server1`, `crypto-producer`, `model-trainer` and `prediction-client`. To create these we can run the following from the project's root directory
```shell
./tools/gcp/create master server0 server1 crypto-producer model-trainer prediction-client
```
or alternatively, if they have already been created but are current terminated, simply start them again with
```shell
./tools/gcp/start master server0 server1 crypto-producer model-trainer prediction-client
```

We also need to download and convert the MNIST dataset on both the model trainer and the prediction client
```shell
gcloud compute ssh model-trainer --command='python3 tf-encrypted/examples/mnist/download.py'
gcloud compute ssh prediction-client --command='python3 tf-encrypted/examples/mnist/download.py'
```

### Launching servers

Once the instances are ready the next step is to link them together by creating and distributing a new configuration file
```shell
./tools/gcp/link master server0 server1 crypto-producer model-trainer prediction-client
```
which will put an updated `config.json` file in the home directory on each instance, following by
```shell
./tools/gcp/serve master server0 server1 crypto-producer model-trainer prediction-client
```
which will launch a TensorFlow server on all of them.

### Running

Finally, with the above in place we can the example using
```shell
gcloud compute ssh master --command='rm -rf /tmp/tensorboard; python3 tf-encrypted/examples/mnist/run.py config.json'
```
that will (optionally) first clear any TensorBoard logs on the master that was previously recorded.

Once completely, the logs may be pulled down from the master
```shell
rm -rf /tmp/tensorboard; gcloud compute scp --recurse master:/tmp/tensorboard /tmp/tensorboard
```
and explored by launching TensorBoard
```shell
tensorboard --logdir=/tmp/tensorboard
```
and navigating to `http://localhost:6006/` in a browser
