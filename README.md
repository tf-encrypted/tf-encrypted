<p align="center">
<img align="center" width="200" height="200" src="./img/logo.png"/>
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

# Contributions

Several people have had an impact on the development of this library (in alphabetical order):

- [Andrew Trask](https://github.com/iamtrask)
- [Koen van der Veen](https://github.com/koenvanderveen)

and several companies have invested significant resources (in alphabetical order):

- [Dropout Labs](https://dropoutlabs.com/) continues to sponsor a large amount of both research and engineering
- [OpenMined](https://openmined.org) was the breeding ground for the initial idea and continues to support discussions and guidance

## Reported uses

Happy to hear all!
