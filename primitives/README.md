## Installation

These commands must be run *in the root directory*, i.e. the parent of this one.

To set up TensorFlow (one time requirement):

```
<<<<<<< HEAD
./configure
=======
pip install -U tensorflow==2.1
./configure.sh
>>>>>>> 08caea9d02a6d6360cbe7b82d7652f24560752ca
```

To build the pip package:

```
bazel build @primitives//:build_pip_package
bazel-bin/external/primitives/build_pip_package artifacts
```

This will deliver the pip package in the `artifacts` directory. To install it

```
pip install -U artifacts/tf_encrypted_primitives-0.0.1-py3-none-any.whl
```

Note that package can be used on its own but does *not* currently work with core TF Encrypted due to difference in TensorFlow version.

To run related tests:

```
bazel test "@primitives//..."
```
