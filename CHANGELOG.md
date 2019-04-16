# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.1]

**Added**

- We now support TensorFlow 1.13.1.  We plan to support to 2.0+ in a future release.
- Am example of performing secure aggregation for federated learning is now available in the `examples/federated-learning/` directory.
- New scripts in `bin/` for launching TF Encrypted jobs on local machines and cloud clusters. We have primarily used these in concert with GCP & GKE.
- The tensor seeding from 0.4.0 has been implemented for all BackingTensor types and protocols.
- A variety of TF Ops have been added to the Converter, including an example of a "special op" (`required_space_to_batch_paddings`) that converts an entire subgraph of an imported Tensorflow graph into TFE instead of converting nodes in the original TensorFlow graph one-by-one. The converter will be generalized and documented much more in a future release.
- New ops include division between a private numerator and public denominator, as well as a secure version of `tf.negative`.
- The `PrivateModel` abstraction can now be tagged.

**Breaking**

- All examples have been updated. Most now use the higher-level Keras interface to TensorFlow before building models in TF Encrypted.
- The various BackingTensor types have been refactored and unified under a set of factories. This affects how Configs and Protocols are instantiated.  Please see the examples for more.

**Changed**

- The `tf_encrypted.layers.Conv2D` layer now automatically includes a bias add. The bias is initialized as a zero tensor if none is provided. This will be handled more elegantly in a future release.

## [0.4.0]

**Added**

- SecureNN with int64 support has landed, a tensorflow build with int64 matmul support must be used. See details on how to do that [here](./README.md#securenn-int64-support)
- Cryptographically secure random numbers feature has been implemented.
- We now send random seeds from the crypto-producer instead of full tensors when masking and computing multiplications. This reduces the amount of data sent across the wire between parties effecting operations such as mul, matmul, conv2d, and square.
- Four new operations are now supported: Pad, BatchToSpaceND, SpaceToBatchND, ArgMax

**Changed**

- There are now separate wheels published to PyPI for MacOS and Linux.
- Various documentation updates.

## [0.3.0]

**Breaking**

- Default naming of crypto producer and weights provider have been changed to `crypto-producer` and `weights-provider` respectively.

**Changed**

- Various documentation updates.

## [0.2.0]

**Breaking**

- Import path renamed from `tensorflow_encrypted` to `tf_encrypted`

**Added**

- Added the beginnings of documentation which is available on [readthedocs](https://tf-encrypted.readthedocs.io/en/latest/)
- Added a CHANGELOG.md file to the project root.
