<img src="https://tinyurl.com/tfelogo" width="65%" />

TF Encrypted is a framework for encrypted machine learning in TensorFlow. It looks and feels like TensorFlow, taking advantage of the ease-of-use of the Keras API while enabling training and prediction over encrypted data via secure multi-party computation and homomorphic encryption. TF Encrypted aims to make privacy-preserving machine learning readily available, without requiring expertise in cryptography, distributed systems, or high performance computing.

See below for more [background material](#background--further-reading), explore the [examples](./examples/), or visit the [documentation](./docs/) to learn more about how to use the library. You are also more than welcome to join our [Slack channel](https://join.slack.com/t/tf-encrypted/shared_invite/enQtNjI5NjY5NTc0NjczLTFkYTRjYWQ0ZWVlZjVmZTVhODNiYTA2ZTdlNWRkMWE4MTI3ZGFjMWUwZDhhYTk1NjJkZTRiZjBhMzMyMjNlZmQ) for all questions around use and development.

[![Website](https://img.shields.io/website/https/tf-encrypted.io.svg)](https://tf-encrypted.io) [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://tf-encrypted.readthedocs.io/en/latest/) [![PyPI](https://img.shields.io/pypi/v/tf-encrypted.svg)](https://pypi.org/project/tf-encrypted/) [![CircleCI Badge](https://circleci.com/gh/tf-encrypted/tf-encrypted/tree/master.svg?style=svg)](https://circleci.com/gh/tf-encrypted/tf-encrypted/tree/master)

# Installation

TF Encrypted is available as a package on [PyPI](https://pypi.org/project/tf-encrypted/) supporting Python 3.5+ and TensorFlow 1.12.0+:

```bash
pip3 install tf-encrypted
```

Alternatively, installing from source can be done using:

```bash
git clone https://github.com/tf-encrypted/tf-encrypted.git
cd tf-encrypted
pip3 install -e .
```

This latter is useful on platforms for which the pip package has not yet been compiled but is also needed for [development](./docs/CONTRIBUTING.md). Note that this will get you a working basic installation, yet a few more steps are required to match the performance and security of the version shipped in the pip package, see the [installation instructions](./docs/INSTALL.md).

# Usage

The following is an example of simple matmul on encrypted data using TF Encrypted:

```python
import tensorflow as tf
import tf_encrypted as tfe

@tfe.local_computation('input-provider')
def provide_input():
    # normal TensorFlow operations can be run locally
    # as part of defining a private input, in this
    # case on the machine of the input provider
    return tf.ones(shape=(5, 10))

# define inputs
w = tfe.define_private_variable(tf.ones(shape=(10,10)))
x = provide_input()

# define computation
y = tfe.matmul(x, w)

with tfe.Session() as sess:
    # initialize variables
    sess.run(tfe.global_variables_initializer())
    # reveal result
    result = sess.run(y.reveal())
```

For more information, check out the [documentation](./docs/) or the [examples](./examples/).

# Roadmap

- High-level APIs for combining privacy and machine learning. So far TF Encrypted is focused on its low-level interface but it's time to figure out what it means for interfaces such as Keras when privacy enters the picture.

- Tighter integration with TensorFlow. This includes aligning with the upcoming TensorFlow 2.0 as well as figuring out how TF Encrypted can work closely together with related projects such as [TF Privacy](https://github.com/tensorflow/privacy) and [TF Federated](https://github.com/tensorflow/federated).

- Support for third party libraries. While TF Encrypted has its own implementations of secure computation, there are other [excellent libraries](https://github.com/rdragos/awesome-mpc/) out there for both secure computation and homomorphic encryption. We want to bring these on board and provide a bridge from TensorFlow.

# Background & Further Reading

The following texts provide further in-depth presentations of the project:

- [Growing TF Encrypted](https://medium.com/dropoutlabs/growing-tf-encrypted-a1cb7b109ab5) outlines the roadmap and motivates why TF Encrypted is a community project
- [Experimenting with TF Encrypted](https://medium.com/dropoutlabs/experimenting-with-tf-encrypted-fe37977ff03c) walks through a simple example of turning an existing TensorFlow prediction model private
- [Private Machine Learning in TensorFlow using Secure Computation](https://arxiv.org/abs/1810.08130) further elaborates on the benefits of the approach, outlines the adaptation of a secure computation protocol, and reports on concrete performance numbers
- [Secure Computations as Dataflow Programs](https://mortendahl.github.io/2018/03/01/secure-computation-as-dataflow-programs/) describes the initial motivation and implementation

# Contributing

Check out our [contribution guide](./docs/CONTRIBUTING.md) for more information, and don't hesitate to send a pull request, open an issue, or ask for help! You can do so either via [GitHub](https://github.com/tf-encrypted/tf-encrypted) or by [joining](https://join.slack.com/t/tf-encrypted/shared_invite/enQtNjI5NjY5NTc0NjczLTFkYTRjYWQ0ZWVlZjVmZTVhODNiYTA2ZTdlNWRkMWE4MTI3ZGFjMWUwZDhhYTk1NjJkZTRiZjBhMzMyMjNlZmQ) our [Slack channel](https://tf-encrypted.slack.com).

[![](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/images/0)](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/links/0)[![](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/images/1)](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/links/1)[![](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/images/2)](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/links/2)[![](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/images/3)](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/links/3)[![](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/images/4)](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/links/4)[![](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/images/5)](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/links/5)[![](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/images/6)](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/links/6)[![](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/images/7)](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/links/7)

## Organizational contributions

We are very grateful for the significant contributions made by the following organizations!

<table>
    <tr>
        <td><a href="https://dropoutlabs.com/"><img src="https://raw.githubusercontent.com/dropoutlabs/files/master/dropout-labs-logo-white-2500.png" alt="Dropout Labs" width="150"/></a></td>
    </tr>
</table>

# Project Status

TF Encrypted is experimental software not currently intended for use in production environments. The focus is on building the underlying primitives and techniques, with some practical security issues postponed for a later stage. However, care is taken to ensure that none of these represent fundamental issues that cannot be fixed as needed.

## Known limitations

- Elements of TensorFlow's networking subsystem does not appear to be sufficiently hardened against malicious users. Proxies or other means of access filtering may be sufficient to mitigate this.

## Support

Please open an [issue](https://github.com/tf-encrypted/tf-encrypted/issues) or feel free to reach out directly on [Slack](https://join.slack.com/t/tf-encrypted/shared_invite/enQtNjI5NjY5NTc0NjczLTFkYTRjYWQ0ZWVlZjVmZTVhODNiYTA2ZTdlNWRkMWE4MTI3ZGFjMWUwZDhhYTk1NjJkZTRiZjBhMzMyMjNlZmQ).

# License

Licensed under Apache License, Version 2.0 (see [LICENSE](./LICENSE) or http://www.apache.org/licenses/LICENSE-2.0). Copyright as specified in [NOTICE](./NOTICE).
