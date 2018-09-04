# TensorFlow Encrypted

This library provides a layer on top of TensorFlow for doing machine learning on encrypted data as initially described in [Secure Computations as Dataflow Programs](https://mortendahl.github.io/2018/03/01/secure-computation-as-dataflow-programs/) and is structured into roughly three levels:

- basic operations for computing on encrypted tensors
- basic machine learning components using these
- ready-to-use models for private machine learning

with the aim of making it easy for researchers and practitioners to experiment in a familiar framework and without being an expert in both machine learning and cryptography.

## Usage

`tf-encrypted` can be imported next to TensorFlow, giving access to extra operations for constructing regular TensorFlow graphs that operate on encrypted data. Since secure computation protocols by nature involve more than one party, each protocol instance takes a list of hostnames as input (three for the `Pond` protocol) and then exposes methods for creating private variables etc. along the lines of traditional TensorFlow programs. Simple wrappers around `tf.Session` can finally be used to execute these.

```python
import tensorflow as tf
import tensorflow_encrypted as tfe

servers = [
    tfe.protocol.Player('10.0.0.1'),
    tfe.protocol.Player('10.0.0.2'),
    tfe.protocol.Player('10.0.0.3')
]

with tfe.protocol.Pond(*servers) as prot:

    w = prot.private_variable(weight_values)
    b = prot.private_variable(bias_values)

    x = prot.private_placeholder(prediction_shape)
    y = prot.sigmoid(prot.dot(x, w) + b)

    res = y.reveal()

with tfe.remote_session() as sess:

    print(res.eval(sess))
```

Note that higher level estimators are also available, such as `tfe.estimator.LogisticClassifier`.

## Installation
We recommend using a package manager like conda.

Dependencies:
- `setuptools`
- `pip`
- `tensorflow`

To install Tensorflow, you can use pip:
```shell
pip install tensorflow
```

If installing the GPU version of Tensorflow, please refer to [their documentation](https://www.tensorflow.org/install/) for how to find the proper build for your platform and CUDA version.  You'll then point your `pip` command with a _tfBinaryURL_ specific to your system, or you'll use the generic:
```shell
pip install tensorflow-gpu
```

Assuming you're in the preferred Python environment, the following code will install `tf-encrypted`:
```shell
git clone https://github.com/mortendahl/tf-encrypted.git
cd tf-encrypted
pip install .[tf] # or .[tf_gpu] if using a tensorflow-gpu build
```
If you're helping with development, use `pip install -e ...` in the last line.

# Examples

<em>(more coming)</em>

## Logistic regression

```python
import tensorflow as tf
import tensorflow_encrypted as tfe

servers = [
    tfe.protocol.Player('10.0.0.1'),
    tfe.protocol.Player('10.0.0.2'),
    tfe.protocol.Player('10.0.0.3')
]

training_providers = [
    tfe.protocol.InputProvider('10.1.1.1'),
    tfe.protocol.InputProvider('10.1.1.2'),
    tfe.protocol.InputProvider('10.1.1.3'),
    tfe.protocol.InputProvider('10.1.1.4'),
    tfe.protocol.InputProvider('10.1.1.5')
]

prediction_providers = [
    tfe.protocol.InputProvider('10.2.2.1'),
    tfe.protocol.InputProvider('10.2.2.2')
]

with tfe.remote_session() as sess:

    with tfe.protocol.Pond(*servers) as prot:

        # instantiate estimator
        logreg = tfe.estimator.LogisticClassifier(
            session=sess,
            num_features=2
        )

        # perform private training based on data from providers
        logreg.prepare_training_data(training_providers)
        logreg.train(epochs=100, batch_size=30)

        # perform private prediction
        prediction = logreg.predict(prediction_providers)
```

# Developing

## Typing

tf-encrypted is developed using types with [mypy](http://mypy-lang.org/).
This means your branch must not produce any errors when the project is run via mypy.
Most popular editors have plugins that allow you to run mypy as you develop so you
can catch problems early.

### Stubs

MyPy uses [`stubs`](http://mypy.readthedocs.io/en/latest/stubs.html#stub-files) to declare
types for external libraries.  There is a standard set of stubs declared [here](https://github.com/python/typeshed)
that should "just work" out of the box.  For external libraries that are not yet
declared in the typeshed, we can define our own or use external dependencies.


#### External

External development happens for types.  We can contribute back to these repositories
as we develop so keep that in mind whenever you need to add some annotations.

The flow to use an external library:
  - fork/clone what you want (fork optional, not necessary unless you will add types)
  - take note of the path you cloned to
  - Update the `MYPYPATH` to include this directory

E.g., for Tensorflow in Atom
  - `git clone https://github.com/partydb/tensorflow-stubs`
  - `cd tensorflow-stubs && pwd | pbcopy # => /Users/bendecoste/code/tensorflow-stubs`
  - add this path to `MYPYPATH`, in atom:
  ![](./img/mypy-external-stub.png)

#### Internal

Create a new folder for the package and add a `__init__.pyi` file

E.g.,

![](./img/mypy-internal-stub.png)

### Atom

In Atom, you can install the [linter-mypy](https://atom.io/packages/linter-mypy) package.

![](./img/invalid-type-atom.png)

### VS Code

In [User Settings](https://code.visualstudio.com/docs/getstarted/settings) add the following information

```json
{
    "python.linting.mypyEnabled": true
}
```

On OSX you may also need to tell VS Code to use python3

```json
{
    "python.pythonPath": "python3"
}
```

You also need to have mypy installed

```
python3 -m pip install mypy
```

After that, you should see errors whenever you develop (if you cause any 😉)

![](./img/invalid-type-vscode.png)

# Contributions

Several people have made contributions to this project in one way or another (in alphabetical order):
- [Andrew Trask](https://github.com/iamtrask)
- [Koen van der Veen](https://github.com/koenvanderveen)
