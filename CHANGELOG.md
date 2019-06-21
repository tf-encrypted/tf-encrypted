# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.6]

**Added**

- More `tfe.keras` functionality to convert `tf.keras` model into `tfe.keras`:
   - `set_weights`
   - `from_config`
   - `model_from_config`
   - `clone_model`
- Example with conversion of `tf.keras` model into `tfe.keras`.
- Improved handling for cases where the secure random operation is not available
- Added methods to the converter to inspect TF and Keras graphs
- `tfe.convert` now supports more than 2 inputs to `tfe.concat`

**Fixed**

- A bug in `tfe.keras.layers.Batchnorm` where the `offset` and `scale` conditions where inverted
- A bug in `tfe.convert` where ops with multiple outputs where not handled properly
- A bug in `tfe.convert` where it couldnt't convert a model correctly when there was more than 
one special ops in the graph

## [0.5.5]

**Added**

- More `tfe.keras` functionality, including BatchNormalization.
- `channels_last` support for `tfe.layers.Batchnorm`
- `tfe.convert` now supports conversions for Ops with multiple output tensors.

**Changed**

- All test files have been moved into the tf_encrypted namespace next to their corresponding functionality in line with the TensorFlow style guide.

**Fixed**

- A bug in `tfe.convert` where tf.split was only returning the first element of the output tensor array.
- An ImportError for users of TF 1.14+.
- A bug where `__radd__` and `__rsub__` were actually computing `__add__` and `__sub__`.
- An overzealous AssertionError in the tfe.serving.QueueServer.

## [0.5.4]

**Added**

- More `tfe.keras` functionality, including `Sequential` model

## [0.5.3]

**Added**

- First steps for the new `tfe.keras` module closely matching that of TensorFlow.
- More examples around private prediction and private training.
- Various notebooks, from getting started to in-depth debugging.
- Decoupled triple generation for Pond and SecureNN, allowing triples to be generated up front or simultaneously with other computations.

**Changed**

- All code is now following the style guide of TensorFlow.

## [0.5.2]

Migration to third party organization, including on [GitHub](https://github.com/tf-encrypted/tf-encrypted/) and [Docker Hub](https://hub.docker.com/r/tfencrypted/tf-encrypted).

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
