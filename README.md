# tf-encrypted

![Status](https://img.shields.io/badge/status-alpha-blue.svg)  [![License](https://img.shields.io/github/license/mortendahl/tf-encrypted.svg)](./LICENSE)  [![PyPI](https://img.shields.io/pypi/v/tf-encrypted.svg)](https://pypi.org/project/tf-encrypted/) [![CircleCI Badge](https://circleci.com/gh/mortendahl/tf-encrypted/tree/master.svg?style=svg)](https://circleci.com/gh/mortendahl/tf-encrypted/tree/master) [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://tf-encrypted.readthedocs.io/en/latest/)

tf-encrypted is a Python library built on top of [TensorFlow](https://www.tensorflow.org) for researchers and practitioners to experiment with privacy-preserving machine learning. It provides an interface similar to that of TensorFlow, and aims at making the technology readily available without first becoming an expert in machine learning, cryptography, distributed systems, and high performance computing.

In particular, the library focuses on:

- **Usability**: The API and its underlying design philosophy make it easy to get started, use, and integrate privacy-preserving technology into pre-existing machine learning processes.
- **Extensibility**: The architecture supports and encourages experimentation and benchmarking of new cryptographic protocols and machine learning algorithms.
- **Performance**: Optimizing for tensor-based applications and relying on TensorFlow's backend means runtime performance comparable to that of specialized stand-alone frameworks.
- **Community**: With a primary goal of pushing the technology forward the project encourages collaboration and open source over proprietary and closed solutions.
- **Security**: Cryptographic protocols are evaluated against strong notions of security and [known limitations](#known-limitations) are highlighted.

See below for more [background material](#background--further-reading) or visit the [documentation](https://tf-encrypted.readthedocs.io/en/latest/index.html) to learn more about how to use the library.

The project has benefitted enormously from the efforts of several contributors following its original implementation, most notably [Dropout Labs](https://dropoutlabs.com/) and members of the [OpenMined](https://www.openmined.org/) community. See below for further [details](#contributions).


# Installation & Usage

tf-encrypted is available as a package on [PyPI](https://pypi.org/project/tf-encrypted/) supporting Python 3.5+ which can be installed using pip:

```bash
$ pip install tf-encrypted
```

The following is an example of simple matmul on encrypted data using tf-encrypted:

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

For more information, check out our full getting started guide in the [documentation](https://tf-encrypted.readthedocs.io/en/latest/usage/getting_started.html).

If you'd like to develop tf-encrypted locally, please read our [contributing guide](./.github/CONTRIBUTING.md#setup).

# SecureNN Int64 Support

To make use of int64 in the SecureNN protocol you'll need to download a special build of tensorflow that contains support for the int64 matrix multiplications. We make no guarantees about these builds and their usage should still be treated as pre-alpha but they make experimenting with int64 possible!

Download for MacOS [here](https://storage.googleapis.com/dropoutlabs-tensorflow-builds/tensorflow-1.9.0-cp35-cp35m-macosx_10_7_x86_64.whl).

Download for Linux [here](https://storage.googleapis.com/dropoutlabs-tensorflow-builds/tensorflow-1.9.0-cp35-cp35m-linux_x86_64.whl).

Now you should just be able to install using pip:

**MacOS**

```
pip install tensorflow-1.9.0-cp35-cp35m-macosx_10_7_x86_64.whl
```

**Linux**

```
pip install tensorflow-1.9.0-cp35-cp35m-linux_x86_64.whl
```

tf-encrypted auto-detects whether int64 support is available or not and uses that by default if so. So no further action will be needed to make use of this cool feature!!

# Background & Further Reading

The following texts provide further in-depth presentations of the project:

- [Secure Computations as Dataflow Programs](https://mortendahl.github.io/2018/03/01/secure-computation-as-dataflow-programs/) describes the initial motivation and implementation
- [Private Machine Learning in TensorFlow using Secure Computation](https://arxiv.org/abs/1810.08130) further elaborates on the benefits of the approach, outlines the adaptation of a secure computation protocol, and reports on concrete performance numbers
- [Experimenting with tf-encrypted](https://medium.com/dropoutlabs/experimenting-with-tf-encrypted-fe37977ff03c) walks through a simple example of turning an existing TensorFlow prediction model private

# Project Status

tf-encrypted is experimental software not currently intended for use in production environments. The focus is on building the underlying primitives and techniques, with some practical security issues post-poned for a later stage. However, care is taken to ensure that none of these represent fundamental issues that cannot be fixed as needed.

## Known limitations

- Elements of TensorFlow's networking subsystem does not appear to be sufficiently hardened against malicious users. Proxies or other means of access filtering may be sufficient to mitigate this.

# Contributing

Don't hesitate to send a pull request, open an issue, or ask for help! Check out our [contribution guide](./.github/CONTRIBUTING.md) for more information!

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
