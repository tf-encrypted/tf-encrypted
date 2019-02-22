# TF Encrypted

![Status](https://img.shields.io/badge/status-alpha-blue.svg)  [![License](https://img.shields.io/github/license/mortendahl/tf-encrypted.svg)](./LICENSE)  [![PyPI](https://img.shields.io/pypi/v/tf-encrypted.svg)](https://pypi.org/project/tf-encrypted/) [![CircleCI Badge](https://circleci.com/gh/mortendahl/tf-encrypted/tree/master.svg?style=svg)](https://circleci.com/gh/mortendahl/tf-encrypted/tree/master) [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://tf-encrypted.readthedocs.io/en/latest/)

TF Encrypted is a Python library built on top of [TensorFlow](https://www.tensorflow.org) for researchers and practitioners to experiment with privacy-preserving machine learning. It provides an interface similar to that of TensorFlow, and aims at making the technology readily available without first becoming an expert in machine learning, cryptography, distributed systems, and high performance computing.

In particular, the library focuses on:

- **Usability**: The API and its underlying design philosophy make it easy to get started, use, and integrate privacy-preserving technology into pre-existing machine learning processes.
- **Extensibility**: The architecture supports and encourages experimentation and benchmarking of new cryptographic protocols and machine learning algorithms.
- **Performance**: Optimizing for tensor-based applications and relying on TensorFlow's backend means runtime performance comparable to that of specialized stand-alone frameworks.
- **Community**: With a primary goal of pushing the technology forward the project encourages collaboration and open source over proprietary and closed solutions.
- **Security**: Cryptographic protocols are evaluated against strong notions of security and [known limitations](#known-limitations) are highlighted.

See below for more [background material](#background--further-reading), explore the [examples](./examples/), or visit the [documentation](./docs/) to learn more about how to use the library.

The project has benefitted enormously from the efforts of several contributors following its original implementation, most notably [Dropout Labs](https://dropoutlabs.com/) and members of the [OpenMined](https://www.openmined.org/) community. See below for further [details](#contributions).

# Installation

TF Encrypted is available as a package on [PyPI](https://pypi.org/project/tf-encrypted/) supporting Python 3.5+ and TensorFlow 1.12.0+ which can be installed using:

```bash
pip3 install tf-encrypted
```

Alternatively, installing from source can be done using:

```bash
git clone https://github.com/mortendahl/tf-encrypted.git
cd tf-encrypted
pip3 install -e .
```

This latter is useful on platforms for which the pip package has not yet been compiled but is also needed for [development](./docs/CONTRIBUTING.md). Note that this will get you a working basic installation, yet a few more steps are required to match the performance and security of the version shipped in the pip package, see the [installation instructions](./docs/INSTALL.md).

## Custom build of TensorFlow

While TF Encrypted will work with the official release of [TensorFlow](https://pypi.org/project/tensorflow/) (version 1.12+), some features currently depend on improvements that have not yet been shipped. In particular, to get speed improvements by using int64 tensors instead of int100 tensors you currently need a custom build of TensorFlow.

Such builds are available for [macOS](https://storage.googleapis.com/dropoutlabs-tensorflow-builds/tensorflow-1.12.0-cp35-cp35m-macosx_10_7_x86_64.whl) and [Linux](https://storage.googleapis.com/dropoutlabs-tensorflow-builds/tensorflow-1.12.0-cp35-cp35m-linux_x86_64.whl) as a temporary solution until the next official release of TensorFlow is out (version 1.13), but no guarantees are made about them and they should be treated as pre-alpha. See more in the [installation instructions](./docs/INSTALL.md#custom-tensorflow).

# Usage

The following is an example of simple matmul on encrypted data using TF Encrypted:

```python
import tensorflow as tf
import tf_encrypted as tfe

def provide_input():
    # local TensorFlow operations can be run locally
    # as part of defining a private input, in this
    # case on the machine of the input provider
    return tf.ones(shape=(5, 10))

# define inputs
w = tfe.define_private_variable(tf.ones(shape=(10,10)))
x = tfe.define_private_input('input-provider', provide_input)

# define computation
y = tfe.matmul(x, w)

with tfe.Session() as sess:
    # initialize variables
    sess.run(tfe.global_variables_initializer())
    # reveal result
    result = sess.run(y.reveal())
```

For more information, check out the [documentation](./docs/) or the [examples](./examples/).


# Background & Further Reading

The following texts provide further in-depth presentations of the project:

- [Secure Computations as Dataflow Programs](https://mortendahl.github.io/2018/03/01/secure-computation-as-dataflow-programs/) describes the initial motivation and implementation
- [Private Machine Learning in TensorFlow using Secure Computation](https://arxiv.org/abs/1810.08130) further elaborates on the benefits of the approach, outlines the adaptation of a secure computation protocol, and reports on concrete performance numbers
- [Experimenting with tf-encrypted](https://medium.com/dropoutlabs/experimenting-with-tf-encrypted-fe37977ff03c) walks through a simple example of turning an existing TensorFlow prediction model private

# Project Status

TF Encrypted is experimental software not currently intended for use in production environments. The focus is on building the underlying primitives and techniques, with some practical security issues postponed for a later stage. However, care is taken to ensure that none of these represent fundamental issues that cannot be fixed as needed.

## Known limitations

- Elements of TensorFlow's networking subsystem does not appear to be sufficiently hardened against malicious users. Proxies or other means of access filtering may be sufficient to mitigate this.

# Contributing

Don't hesitate to send a pull request, open an issue, or ask for help! Check out our [contribution guide](./docs/CONTRIBUTING.md) for more information!

Several individuals have already had an impact on the development of this library (in alphabetical order):

- [Ben DeCoste](https://github.com/bendecoste) (Dropout Labs)
- [Yann Dupis](https://github.com/yanndupis) (Dropout Labs)
- [Morgan Giraud](https://github.com/morgangiraud) (while at Dropout Labs)
- [Ian Livingstone](https://github.com/ianlivingstone) (Dropout Labs)
- [Jason Mancuso](https://github.com/jvmancuso) (Dropout Labs)
- [Justin Patriquin](https://github.com/justin1121) (Dropout Labs)
- [Andrew Trask](https://github.com/iamtrask) (OpenMined)
- [Koen van der Veen](https://github.com/koenvanderveen) (OpenMined)

and several companies have invested significant resources:

- [Dropout Labs](https://dropoutlabs.com/) continues to sponsor a large amount of both research and engineering
- [OpenMined](https://openmined.org) was the breeding ground for the initial idea and continues to support discussions and guidance

# License

Licensed under Apache License, Version 2.0 (see [LICENSE](./LICENSE) or http://www.apache.org/licenses/LICENSE-2.0). Copyright as specified in [NOTICE](./NOTICE).
