<p align="center">
<img align="center" width="200" height="200" src="./img/logo_scatter.png"/>
</p>

# tf-encrypted

![Status](https://img.shields.io/badge/status-alpha-blue.svg)  [![License](https://img.shields.io/github/license/mortendahl/tf-encrypted.svg)](./LICENSE)  [![PyPI](https://img.shields.io/pypi/v/tf-encrypted.svg)](https://pypi.org/project/tf-encrypted/) [![CircleCI Badge](https://circleci.com/gh/mortendahl/tf-encrypted/tree/master.svg?style=svg)](https://circleci.com/gh/mortendahl/tf-encrypted/tree/master) [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://tf-encrypted.readthedocs.io/en/latest/)

This library provides a layer on top of TensorFlow for doing machine learning on encrypted data as initially described in [Secure Computations as Dataflow Programs](https://mortendahl.github.io/2018/03/01/secure-computation-as-dataflow-programs/), with the aim of making it easy for researchers and practitioners to experiment with private machine learning using familiar tools and without being an expert in both machine learning and cryptography. To this end the code is structured into roughly three modules:

- secure operations for computing on encrypted tensors
- basic machine learning operations built on top of these
- ready-to-use components for private prediction and training

that are all exposed through Python interfaces, and all resulting in ordinary TensorFlow graphs for easy integration with other TensorFlow mechanisms and efficient execution.

Several contributors have put resources into the development of this library, most notably [Dropout Labs](https://dropoutlabs.com/) and members of the [OpenMined](https://www.openmined.org/) community (see below for [details](#contributions)).

**Important**: this is experimental software that should not be used in production for security reasons.

## Usage

The following secure computation calculates the average of private inputs from a group of *inputters* running on different machines. Two *servers* and a *crypto producer* are doing the actual computation on encrypted data, with only the *result receiver* being able to decrypt the final result.

```python
# get named players from hostmap configuration
server0 = config.get_player('server0')
server1 = config.get_player('server1')
crypto_producer = config.get_player('crypto_producer')

# perform secure operations using the Pond protocol
with tfe.protocol.Pond(server0, server1, crypto_producer) as prot:

    # get input from inputters as private values
    inputs = [prot.define_private_input(inputter) for inputter in inputters]

    # sum all inputs and multiply by count inverse (ie divide)
    result = reduce(lambda x, y: x + y, inputs) * (1 / len(inputs))

    # send result to receiver who can finally decrypt
    result_op = prot.define_output([result], result_receiver)

    with tfe.Session() as sess:
        tfe.run(sess, result_op, tag='average')
```

To get this running we first have to import `tf_encrypted`, and since we also want to use ordinary Tensorflow locally on both the inputters and the result receiver we also import `tensorflow`.

```python
import tensorflow as tf
import tf_encrypted as tfe
```

Furthermore, we also need to specify the *local* behaviour of the inputters and the receiver by defining classes deriving from respectively `tfe.io.InputProvider` and `tfe.io.OutputReceiver`.

```python
class Inputter(tfe.io.InputProvider):
    def provide_input(self) -> tf.Tensor:
        # use TensorFlow to pick random tensor as this player's input
        return tf.random_normal(shape=(10,))

class ResultReceiver(tfe.io.OutputReceiver):
    def receive_output(self, average:tf.Tensor) -> tf.Operation:
        # when value is received here it has already been decrypted locally
        return tf.Print([], [average], summarize=10, message="Average:")

inputters = [
    Inputter(config.get_player('inputter-0')),
    Inputter(config.get_player('inputter-1')),
    Inputter(config.get_player('inputter-2')),
    Inputter(config.get_player('inputter-3')),
    Inputter(config.get_player('inputter-4'))
]

result_receiver = ResultReceiver(config.get_player('result_receiver'))
```

Finally, we also loaded the pre-specified hostmap configuration from file using.

```python
# load host map configuration from file
config = tfe.config.load('config.json')
```

Take a look at [`/tools/gcp/link`](./tools/gcp/link) as an example to generate the config file for gcp. If you run it locally, you can use simply `tfe.LocalConfig`. You can find an example [here](./examples/federated-average/run.py).

See [`examples/federated-average/`](./examples/federated-average/) for ready-to-run code and further details, and see the [`examples`](./examples/) directory for additional and more advanced examples.

## Installation

To install the library simply run the following from within your preferred Python environment:

```shell
git clone https://github.com/mortendahl/tf-encrypted.git
cd tf-encrypted
pip install -e .
```

Note however that currently **only Python 3.5 and 3.6 are supported**; to manage this we recommend using a package manager like pip or conda.

After successful installation you should be able to e.g. run the examples

```shell
python3 examples/federated-average/run.py
```

using a local configuration.

### Google Cloud Platform

Please see [`tools/gcp/`](./tools/gcp/) for further information about setting up and running on the Google Cloud Platform.

### Development

Please see [`DEVELOP`](./DEVELOP.md) for guidelines and further instructions for setting up the project for development.

# License

Licensed under Apache License, Version 2.0 (see [LICENSE](./LICENSE) or http://www.apache.org/licenses/LICENSE-2.0). Copyright as specified in [NOTICE](./NOTICE).

# Contributions

Several people have had an impact on the development of this library (in alphabetical order):

- [Andrew Trask](https://github.com/iamtrask)
- [Koen van der Veen](https://github.com/koenvanderveen)

and several companies have invested significant resources (in alphabetical order):

- [Dropout Labs](https://dropoutlabs.com/) continues to sponsor a large amount of both research and engineering
- [OpenMined](https://openmined.org) was the breeding ground for the initial idea and continues to support discussions and guidance

## Reported uses

Happy to hear all!
