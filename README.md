<img src="https://tinyurl.com/tfelogo" width="65%" />

TF Encrypted is a framework for encrypted machine learning in TensorFlow. It looks and feels like TensorFlow, taking advantage of the ease-of-use of the Keras API while enabling training and prediction over encrypted data via secure multi-party computation and homomorphic encryption. TF Encrypted aims to make privacy-preserving machine learning readily available, without requiring expertise in cryptography, distributed systems, or high performance computing.

See below for more [background material](#background--further-reading), explore the [examples](./examples/), or visit the [documentation](./docs/) to learn more about how to use the library. 

[![Website](https://img.shields.io/website/https/tf-encrypted.io.svg)](https://tf-encrypted.io) [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://tf-encrypted.readthedocs.io/en/latest/) [![PyPI](https://img.shields.io/pypi/v/tf-encrypted.svg)](https://pypi.org/project/tf-encrypted/) [![CircleCI Badge](https://circleci.com/gh/tf-encrypted/tf-encrypted/tree/master.svg?style=svg)](https://circleci.com/gh/tf-encrypted/tf-encrypted/tree/master)

# Installation

TF Encrypted is available as a package on [PyPI](https://pypi.org/project/tf-encrypted/) supporting Python 3.5+ and TensorFlow 1.12.0+:

```bash
pip install tf-encrypted
```

Creating a conda environment to run TF Encrypted code can be done using:

```
conda create -n tfe python=3.6
conda activate tfe
conda install tensorflow notebook
pip install tf-encrypted
```

Alternatively, installing from source can be done using:

```bash
git clone https://github.com/tf-encrypted/tf-encrypted.git
cd tf-encrypted
pip install -e .
make build
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

# Performance

All tests are performed by using the ABY3 protocol among 3 machines, each with 4 cores (Intel Xeon Platinum 8369B CPU @ 2.70GHz). The LAN environment has a bandwidth of 40 Gbps and a RTT of 0.02 ms, and the WAN environment has a bandwidth of 352 Mbps and a RTT of 40 ms.

You can find source code of the following benchmarks in [`./examples/benchmark/`](./examples/benchmark/) and corresponding guidelines of how to reproduce them.

## Benchmark 1: Sort and Max

Graph building is a one-time cost, while LAN or WAN timings are average running time across multiple runs. For example, it takes 58.63 seconds to build the graph for Resnet50 model, and afterwards, it only takes 4.742 seconds to predict each image.

|                                        | Build graph<br/>(seconds) | LAN<br/>(seconds) | WAN<br/>(seconds) |
| -------------------------------------- | ------------------------- | ----------------- | ----------------- |
| Sort/Max (1,000)<sup>1</sup>           | 0.90                      | 0.13              | 11.51             |
| Sort/Max (1,000,000)<sup>1</sup>       | 74.70                     | 117.451           | 1133.00           |
| Max (1,000 $\times$ 4)<sup>2</sup>     | 2.02                      | 0.01              | 0.51              |
| Max (1,000,000 $\times$ 4)<sup>2</sup> | 2.05                      | 3.66              | 15.28             |

<sup>1</sup> `Max` is implemented by using a sorting network, hence its performance is essentially the same as `Sort`. Sorting network can be efficiently constructed as a TF graph. The traditional way of computing `Max` by using a binary comparison tree does not work well in a TF graph, because the graph becomes huge when the number of elements is large.

<sup>2</sup> This means 1,000 (respectively, 1,000,000) invocations of `max` on 4 elements, which is essentially a `MaxPool` with pool size of `2 x 2`.

## Benchmark 2: Neural Network Inference

We show the strength of TFE by loading a normal TF model (RESNET50) and run private inference on top of it.

|                                   | Build graph<br/> | LAN<br/>          | WAN<br/> |
| --------------------------------- | ---------------- | ----------------- | -------- |
| RESNET50 inference time (seconds) | 57.79            | 13.55<sup>1</sup> | 126.89   |

<sup>1</sup> This is currently one of the fastest implementation of secure RESNET50 inference (three-party). Comparable with [CryptGPU](https://eprint.iacr.org/2021/533) , [SecureQ8](https://eprint.iacr.org/2019/131), and faster than [CryptFLOW](https://arxiv.org/abs/1909.07814).

## Benchmark 3: Neural Network Training

We benchmark the performance of training several neural network models on the MNIST dataset (60k training images, 10k test images, and batch size is 128). The definitions of these models can be found in [`examples/benchmark/mnist/private_network_training.py`](examples/benchmark/mnist/private_network_training.py).

We compare the performance with another highly optimized MPC library [MP-SPDZ](https://github.com/data61/MP-SPDZ).

|             | Accuracy (epochs) | Accuracy (epochs) | Seconds per Batch (LAN) | Seconds per Batch (LAN) | Seconds per Batch (WAN) | Seconds per Batch (WAN) |
|:----------- |:-----------------:|:-----------------:|:-----------------------:|:-----------------------:|:-----------------------:|:-----------------------:|
|             | MP-SPDZ           | TFE               | MP-SPDZ                 | TFE                     | MP-SPDZ                 | TFE                     |
| A (SGD)     | 96.7% (5)         | 96.8% (5)         | 0.098                   | 0.138                   | 9.724                   | 5.075                   |
| A (AMSgrad) | 97.8% (5)         | 97.3% (5)         | 0.228                   | 0.567                   | 21.038                  | 17.780                  |
| A (Adam )   | 97.4% (5)         | 97.3% (5)         | 0.221                   | 0.463                   | 50.963                  | 16.958                  |
| B (SGD)     | 97.5% (5)         | 98.7% (5)         | 0.571                   | 4.000                   | 60.755                  | 25.300                  |
| B (AMSgrad) | 98.6% (5)         | 99.0% (5)         | 0.680                   | 4.170                   | 71.983                  | 28.424                  |
| B (Adam)    | 98.8% (5)         | 98.8% (5)         | 0.772                   | 4.075                   | 98.108                  | 28.184                  |
| C (SGD)     | 98.5% (5)         | 98.8% (5)         | 1.175                   | 6.223                   | 91.341                  | 37.678                  |
| C (AMSgrad) | 98.9% (5)         | 99.0% (5)         | 1.568                   | 7.336                   | 119.271                 | 83.695                  |
| C (Adam)    | 99.0% (5)         | 99.1% (5)         | 2.825                   | 6.858                   | 195.013                 | 81.275                  |
| D (SGD)     | 97.6% (5)         | 97.5% (5)         | 0.134                   | 0.355                   | 15.083                  | 6.112                   |
| D (AMSgrad) | 98.4% (5)         | 98.1% (5)         | 0.228                   | 0.682                   | 26.099                  | 17.063                  |
| D (Adam)    | 98.2% (5)         | 98.0% (5)         | 0.293                   | 0.605                   | 54.404                  | 16.190                  |

We also give the performance of training a logistic regression model in the following table. This model is trained to classify two classes: small digits (0-4) vs large digits (5-9). Details can be found in [`examples/benchmark/mnist/private_lr_training.py`](examples/benchmark/mnist/private_lr_training.py)

|              | Accuracy (epochs) | Seconds per Batch (LAN) | Seconds per Batch (WAN) |
|:------------ |:-----------------:|:-----------------------:|:-----------------------:|
| LR (SGD)     | 84.1% (5)         | 0.012                   | 0.760                   |
| LR (AMSgrad) | 85.5% (5)         | 0.025                   | 1.567                   |
| LR (Adam)    | 85.8% (5)         | 0.021                   | 1.353                   |

# Roadmap

- High-level APIs for combining privacy and machine learning. So far TF Encrypted is focused on its low-level interface but it's time to figure out what it means for interfaces such as Keras when privacy enters the picture.

- Tighter integration with TensorFlow. This includes aligning with the upcoming TensorFlow 2.0 as well as figuring out how TF Encrypted can work closely together with related projects such as [TF Privacy](https://github.com/tensorflow/privacy) and [TF Federated](https://github.com/tensorflow/federated).

- Support for third party libraries. While TF Encrypted has its own implementations of secure computation, there are other [excellent libraries](https://github.com/rdragos/awesome-mpc/) out there for both secure computation and homomorphic encryption. We want to bring these on board and provide a bridge from TensorFlow.

<img src="https://raw.githubusercontent.com/tf-encrypted/assets/master/app-stack.png" width="55%" />

# Background & Further Reading

Blog posts:

- [Introducing TF Encrypted](https://alibaba-gemini-lab.github.io/docs/blog/tfe/) walks through a simple example showing two data owners jointly training a logistic regression model using TF Encrypted on a vertically split dataset (*by Alibaba Gemini Lab*)

- [Federated Learning with Secure Aggregation in TensorFlow](https://medium.com/dropoutlabs/federated-learning-with-secure-aggregation-in-tensorflow-95f2f96ebecd) demonstrates using TF Encrypted for secure aggregation of federated learning in pure TensorFlow (*by Justin Patriquin at Cape Privacy*)

- [Encrypted Deep Learning Training and Predictions with TF Encrypted Keras](https://medium.com/dropoutlabs/encrypted-deep-learning-training-and-predictions-with-tf-encrypted-keras-557193284f44) introduces and illustrates first parts of our encrypted Keras interface (*by Yann Dupis at Cape Privacy*)

- [Growing TF Encrypted](https://medium.com/dropoutlabs/growing-tf-encrypted-a1cb7b109ab5) outlines the roadmap and motivates TF Encrypted as a community project (*by Morten Dahl*)

- [Experimenting with TF Encrypted](https://medium.com/dropoutlabs/experimenting-with-tf-encrypted-fe37977ff03c) walks through a simple example of turning an existing TensorFlow prediction model private (*by Morten Dahl and Jason Mancuso at Cape Privacy*)

- [Secure Computations as Dataflow Programs](https://mortendahl.github.io/2018/03/01/secure-computation-as-dataflow-programs/) describes the initial motivation and implementation (*by Morten Dahl*)

Papers:

- [Privacy-Preserving Collaborative Machine Learning on Genomic Data using TensorFlow](https://arxiv.org/abs/2002.04344) outlines the [iDASH'19](http://www.humangenomeprivacy.org/2019/) winning solution built on TF Encrypted (*by Cheng Hong, et al.*)

- [Crypto-Oriented Neural Architecture Design](https://arxiv.org/abs/1911.12322) uses TF Encrypted to benchmark ML optimizations made to better support the encrypted domain (*by Avital Shafran, Gil Segev, Shmuel Peleg, and Yedid Hoshen*)

- [Private Machine Learning in TensorFlow using Secure Computation](https://arxiv.org/abs/1810.08130) further elaborates on the benefits of the approach, outlines the adaptation of a secure computation protocol, and reports on concrete performance numbers (*by Morten Dahl, Jason Mancuso, Yann Dupis, et al.*)

Presentations:

- [Privacy-Preserving Machine Learning with TensorFlow](https://github.com/capeprivacy/tf-world-tutorial), TF World 2019 (*by Jason Mancuso and Yann Dupis at Cape Privacy*); see also the [slides](https://github.com/capeprivacy/tf-world-tutorial/blob/master/TensorFlow-World-Tutorial-2019-final.pdf)

- [Privacy-Preserving Machine Learning in TensorFlow with TF Encrypted](https://conferences.oreilly.com/artificial-intelligence/ai-ny-2019/public/schedule/detail/76542), O'Reilly AI 2019 (*by Morten Dahl at Cape Privacy*); see also the [slides](https://github.com/mortendahl/talks/blob/master/OReillyAI19-slides.pdf)

Other:

- [Privacy Preserving Deep Learning – PySyft Versus TF Encrypted](https://blog.exxactcorp.com/privacy-preserving-deep-learning-pysyft-tfencrypted/) makes a quick comparison between PySyft and TF Encrypted, correctly hitting on our goal of being the encryption backend in PySyft for TensorFlow (*by Exxact*)

- [Bridging Microsoft SEAL into TensorFlow](https://medium.com/dropoutlabs/bridging-microsoft-seal-into-tensorflow-b04cc2761ad4) takes a first step towards integrating the Microsoft SEAL homomorphic encryption library and some of the technical challenges involved (*by Justin Patriquin at Cape Privacy*)

# Development and Contribution

TF Encrypted is open source community project developed under the Apache 2 license and maintained by a set of core developers. We welcome contributions from all individuals and organizations, with further information available in our [contribution guide](./docs/CONTRIBUTING.md). We invite any organizations interested in [partnering](#organizational-contributions) with us to reach out via [email](mailto:contact@tf-encrypted.io).

Don't hesitate to send a pull request, open an issue, or ask for help! We use [ZenHub](https://www.zenhub.com/extension) to plan and track GitHub issues and pull requests.

## Individual contributions

We appreciate the efforts of [all contributors](https://github.com/tf-encrypted/tf-encrypted/graphs/contributors) that have helped make TF Encrypted what it is! Below is a small selection of these, generated by [sourcerer.io](https://sourcerer.io/) from most recent stats:

[![](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/images/0)](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/links/0)[![](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/images/1)](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/links/1)[![](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/images/2)](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/links/2)[![](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/images/3)](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/links/3)[![](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/images/4)](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/links/4)[![](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/images/5)](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/links/5)[![](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/images/6)](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/links/6)[![](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/images/7)](https://sourcerer.io/fame/mortendahl/tf-encrypted/tf-encrypted/links/7)

## Organizational contributions

We are very grateful for the significant contributions made by the following organizations!

<table>
    <tr>
        <td><a href="https://capeprivacy.com/"><img src="https://raw.githubusercontent.com/tf-encrypted/assets/master/other/capeprivacy-logo.png" alt="Cape Privacy" width="150"/></a></td>
        <td><a href="https://www.alibabagroup.com/"><img src="https://raw.githubusercontent.com/tf-encrypted/assets/master/other/alibaba-logo.png" alt="Alibaba Security Group" width="150"/></a></td>
        <td><a href="https://openmined.org/"><img src="https://raw.githubusercontent.com/tf-encrypted/assets/master/other/openmined-logo.png" alt="OpenMined" width="150"/></a></td>
    </tr>
</table>

# Project Status

TF Encrypted is experimental software not currently intended for use in production environments. The focus is on building the underlying primitives and techniques, with some practical security issues postponed for a later stage. However, care is taken to ensure that none of these represent fundamental issues that cannot be fixed as needed.

## Known limitations

- Elements of TensorFlow's networking subsystem does not appear to be sufficiently hardened against malicious users. Proxies or other means of access filtering may be sufficient to mitigate this.

## Support

Please open an [issue](https://github.com/tf-encrypted/tf-encrypted/issues), or send an email to [contact@tf-encrypted.io](mailto:contact@tf-encrypted.io).

# License

Licensed under Apache License, Version 2.0 (see [LICENSE](./LICENSE) or http://www.apache.org/licenses/LICENSE-2.0). Copyright as specified in [NOTICE](./NOTICE).
