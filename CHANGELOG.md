# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


## [0.8.0-rc0]
**Added**
- Graph conversion from native TF graph to TFE graph (Resnet50 tested)
- Auto backward propagation for neural network model training
- Various necessary functions for neural network training in the ABY3 protocol
- 3PC Benchmark

**Fixed**
- Compatibility with tf 1.13.2


## [0.7.0]
**Fixed**

- Fix a buggy CI workflow that fails to use cache in 'deploy' step (because of a different image from 'build step')
- Remove skipping conditions in aby3 test

**Added**
- ABY3 available on pypi


## [0.5.9]

**Added**

- We now support TensorFlow 1.14.0
- OpenMined as organizational contributor in the README
- Links to TF Seal and Keras blog posts in the README


**Changed**

- Add more detailed explanations to the Keras Classification notebooks

**Fixed**

- Public division by adding reciprotal (tf.math.reciprotal) support
- A bug in interactive truncation which could have lead to overflow
- SecureNN's ReLU out of memory error for large tensors. Above a certain threshold (tensor size), the tensor gets splitted. ReLU is performed on each split and the results are then concatenated.
- An edge case in `tfe.convert` where it couldnt't convert a model correctly using specific special ops in the graph


## [0.5.8]

**Added**

- Converter support for DepthwiseConv2d layer
- `tfe.keras.GlobalAveragePooling2D`: now support global average pooling operation for spatial data
- `tfe.keras.GlobalMaxPooling2D`: now supports global max pooling operation for spatial data
- Added training support basic models:
  - Backpropagation for `tfe.keras.layers.dense` and sigmoid activation
   - `tfe.keras.losses.BinaryCrossentropy`: now supports binary cross-entropy loss.
   - `tfe.keras.optimizers.SGD`: now supports stochastic gradient descent optimizer. 
   - Added `compile` method to `tfe.keras.models.Sequential` to configure the model for training.
   - Added `fit` method to `tfe.keras.models.Sequential` to trains models for a given number of epochs.
 - Bloom's example for fast linear regression
 - Add example of Keras model training with differential privacy, combined with predictions on encrypted data.

**Changed**

- More detailed error message when raising error for unknown layer arguments.

**Fixed**
- Explicitly use int64 for numpy in Pond protocol because `int` is interpreted differently on Windows (32bits) and macOS (64 bits).
- Converter raises exception when passing empty model.

## [0.5.7]

**Added**

- `tfe.local_computation` decorator, which is now the preferred way for providing inputs/outputs to a secure computation in TFE. All of the examples have been updated with example usage.
- `tfe.keras.layers.DepthwiseConv2D` -- converter support for the layer will land in the next release.
- More ops supported by the converter, including `SplitV` and `tf.keras.layers.BatchNormalization`
- `set_weights` for tfe.keras models and layers now accepts private variables as well as numpy arrays.
- `secure_model` now supports batch predictions.

**Changed**

- Examples now use the tfe.keras API for building models instead of lower level tfe Ops.
- Documentation is now generated from Google-style docstrings. As a result, we are only building docs for the tfe.keras API. Docstrings for other modules will be converted to Google-style and published progressively.

**Fixed**
- `tfe.keras` layers will now check to see if defaults have been changed from their originals in tf.keras, and surface errors whenever modified kwargs aren't supported. Some layers were failing to instantiate because these checks were too specific.

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
