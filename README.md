<p align="center">
<img align="center" width="200" height="200" src="./img/logo.png"/>
</p>

# tf-encrypted

![Status](https://img.shields.io/badge/status-alpha-blue.svg)  [![License](https://img.shields.io/github/license/mortendahl/tf-encrypted.svg)](./LICENSE)  [![PyPI](https://img.shields.io/pypi/v/tf-encrypted.svg)](https://pypi.org/project/tf-encrypted/) [![CircleCI Badge](https://circleci.com/gh/mortendahl/tf-encrypted/tree/master.svg?style=svg)](https://circleci.com/gh/mortendahl/tf-encrypted/tree/master) [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://tf-encrypted.readthedocs.io/en/latest/)

tf-encrypted is a Python library built on top of [TensorFlow](https://www.tensorflow.org) focused on making it easy for researchers and practitioners to experiment with privacy-preserving machine learning without needing to be an expert in machine learning, cryptography, distributed systems, or high performance computing. For more information on performance and security please read the [Private Machine Learning in TensorFlow using Secure Computation](link here) paper.

To achieve its vision, this library has several goals:

- **Simple**: It should be easy to get started, experiment with, use, benchmark, and deeply integrate with pre-existing data science workflows.
- **Extensible**: Extending tf-encrypted to experiment with new protocols, tensor implementations, and machine learning algorithms should be a first class citizen.
- **Performant**: All of the fastest known implementations of protocols, tensor implementations, and machine learning algorithms in a private setting should exist within this library.
- **Secure**: A user should have to go out of their way to use the library in a non-secure, non-private fashion. New protocols are marked as experimental until they've met our high bar of security.
- **Community-Driven**: Community-first as a decision making framework. We can only achieve these goals by building a community of researchers, contributors, and users around the project.

Learn more about how to use the project by visiting the [documentation](https://tf-encrypted.readthedocs.io/en/latest/index.html).

Several contributors have put resources into the development of this library, most notably [Dropout Labs](https://dropoutlabs.com/) and members of the [OpenMined](https://www.openmined.org/) community (see below for [details](#contributions)). The approached used by this library was first described by Morten Dahl in his [Secure Computations as Dataflow Programs](https://mortendahl.github.io/2018/03/01/secure-computation-as-dataflow-programs/)) blog post.

# Installation & Usage

tf-encrypted is available as a package on [pypi.org](https://pypi.org/project/tf-encrypted/) which can be installed using pip:

```
$ pip install tf-encrypted
```

The following is an example of simple matmul on encrypted data using tf-encrypted:

```python
import numpy as np
import tf_encrypted as tfe

a = np.ones((10,10))
x = tfe.define_private_variable(a)
y = x.matmul(x)

with tfe.Session() as sess:
    sess.run(tfe.global_variables_initializer(), tag='init')
    actual = sess.run(y.reveal(), tag='reveal')
```

For more information, checkout our full getting started guide in our [documentation](https://tf-encrypted.readthedocs.io/en/latest/usage/getting_started.html)!

# Project Status

tf-encrypted is experimental software that is not ready for use in any production environment for security reasons. We're currently focused on building the underlying primitives that enabled cryptographist, machine learning researchers, and data scientists to experiment with private machine learning.

We are actively seeking outside contributions to help us move from experimental to production-ready software. Don't hesitate to send a pull request or open an issue, we'd love to work with anyone interested in democratizing private machine learning.

# License

Licensed under Apache License, Version 2.0 (see [LICENSE](./LICENSE) or http://www.apache.org/licenses/LICENSE-2.0). Copyright as specified in [NOTICE](./NOTICE).

# Contributions

Several people have had an impact on the development of this library (in alphabetical order):

- [Andrew Trask](https://github.com/iamtrask)
- [Koen van der Veen](https://github.com/koenvanderveen)

and several companies have invested significant resources (in alphabetical order):

- [Dropout Labs](https://dropoutlabs.com/) continues to sponsor a large amount of both research and engineering
- [OpenMined](https://openmined.org) was the breeding ground for the initial idea and continues to support discussions and guidance
