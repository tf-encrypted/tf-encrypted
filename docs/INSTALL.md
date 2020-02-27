# Installation

The most convenient way of installing TF Encrypted is via [the official pip package](https://pypi.org/project/tf-encrypted/):

```
pip3 install tf-encrypted
```

However, for development or on certain platforms this is not sufficient. In these cases a source code installation is more fitting as described in this document. If you encounter any problems during installation we suggest taking a look at the [complete instructions](#complete-instructions) given towards the end of this document.

**Important:** For the rest of this document we assume that `pip` refers to version 3, which will be the case if you are using a virtual environment such as [conda](https://conda.io/) or [virtualenv](https://virtualenv.pypa.io/). The steps for a [basic installation](#basics) can be completed without this assumptions by simply using `pip3` instead, however the [`Makefile`](./Makefile) used for [building custom ops](#custom-ops) is written with a virtual environment in mind and will hence require a bit of adaptation.

## Basics

All you need to get started is to have Python 3.5 or 3.6 installed along with [pip](https://pypi.org/project/pip/). The first step is then to clone the source code available on [GitHub](https://github.com/tf-encrypted/tf-encrypted):

```
./ $ git clone https://github.com/tf-encrypted/tf-encrypted.git
./ $ cd tf-encrypted
```

and install it as a pip package. The simplest way of doing this is by running:

```
(venv) ./tf-encrypted/ $ pip install -e .
```

which will automatically install all needed dependencies, most notably [TensorFlow](https://www.tensorflow.org/install/) version 1.12 or above; if for some reason this is not desired then use:

```
(venv) ./tf-encrypted/ $ pip install -e . --no-deps
```

instead and check that you have all dependencies installed if you encounter any problems later.

That's it - you should now have a working copy ready for development!

Running code at this point may however generate warnings related to sub-optimal performance and security. The reason for this is twofold:

- some features rely on [custom ops](https://www.tensorflow.org/guide/extend/op) that first needs to be compiled;
- some features have not yet shipped as part of the official TensorFlow distribution.

We address both below but stress that they can be skipped for initial experiments.

## Custom Ops

Certains operations, such as secure randomness generation, rely on C++ extensions of TensorFlow known as [custom ops](https://www.tensorflow.org/guide/extend/op). These come precompiled with the [pip package](https://pypi.org/project/tf-encrypted/) but need to be manually compiled when installing from source code as done above.

On macOS this is straight forward once libtool and automake are installed (see the [detailed instructions](#complete-instructions)):

```
(venv) ./tf-encrypted/ $ make build
```

however on Debian and Ubuntu this can cause some issues as described next.

### Debian and Ubuntu

For these platforms we recommend building custom ops in the [official docker container](https://github.com/tensorflow/custom-op) to avoid ABI compatibility issues. First, pull down the docker image:

```
./tf-encrypted/ $ sudo docker pull tensorflow/tensorflow:custom-op
```

Once finished, run a shell in the container:

```
./tf-encrypted/ $ sudo docker run -it -v `pwd`:/opt/tf_encrypted \
                  -w /opt/tf_encrypted \
                  tensorflow/tensorflow:custom-op /bin/bash
```

install TensorFlow inside it:

```
(docker) /opt/tf_encrypted $ pip install tensorflow
```

and finally build the actual custom op:

```
(docker) /opt/tf_encrypted $ make build
```

You then can exit docker at this point.

## Testing

To run unit tests as part of development you need to run:

```
(venv) ./tf-encrypted/ $ make test
```

after making sure flake8 is installed:

```
(venv) ./tf-encrypted/ $ pip install flake8
```

## Custom TensorFlow

TF Encrypted officially supports TensorFlow 1.13.1 but if you have a need to run on 1.12.0 and want to take advantage of the int64 tensor speed improvements you'll have to make use of a custom build.

We provide such builds as a temporary solution and no guarantees are made about them and they should be treated as experimental:

- [macOS](https://storage.googleapis.com/dropoutlabs-tensorflow-builds/tensorflow-1.12.0-cp35-cp35m-macosx_10_7_x86_64.whl) <small>(sha256: <tt>734b7c1efd0afa09da1ac22c45be04c89ced3edf203b42dead8fa842b38c278e</tt>)</small>
- [Linux](https://storage.googleapis.com/dropoutlabs-tensorflow-builds/tensorflow-1.12.0-cp35-cp35m-linux_x86_64.whl) <small>(sha256: <tt>5cd9d36f7fdee0b8d8367aa4aa95a1244c09c8dba87ebb4ccff9631058f57c1f</tt>)</small>

Alternatively you can build TensorFlow on your own by e.g. following the [official instructions](https://www.tensorflow.org/install/source).

In both cases should you end up with a wheel file that you can install using pip:

```
(venv) ./ $ pip install tensorflow-1.13.0-XXX.whl
```

TF Encrypted auto-detects which features are available so no further actions are needed.

## Complete Instructions

### macOS

These steps have been tested on macOS Mojave (10.14).

Assuming [conda](https://conda.io/) is already installed we first create a new virtual environment:

```
./ $ conda create --name venv python=3.6 -y
```

Note that TensorFlow currently requires Python version 3.5 or 3.6, and cannot run on the 3.7 which may the default version of Python 3 installed on macOS. [virtualenv](https://virtualenv.pypa.io/) can of course be used as well, but because of this Python version requirement we generally prefer conda.

We can then activate it:

```
./ $ source activate venv
```

and follow the [basic instructions](#basics).

To also be able to compile custom ops we need to make sure a few system pakcages are available; these can be installed using [Homebrew](https://brew.sh/) as follows:

```
./ $ brew install libtool automake git curl
```

Once complete follow these [instructions](#custom-ops).

### Debian and Ubuntu

These steps have been tested on Debian 9, Ubuntu 16.04, and Ubuntu 18.04. See below for comment on [Raspberry Pi](#raspberry-pi).

First install the system tools needed for basic installations:

```
./ $ sudo apt update
./ $ sudo apt upgrade -y
./ $ sudo apt install -y python3-pip virtualenv git
```

then create a virtual environment (on Ubuntu 16.04 we needed to `export LC_ALL=C` first):

```
./ $ virtualenv -p python3 venv
./ $ source venv/bin/activate
```

and follow the [basic instructions](#basics).

To compile custom ops first install docker

```
./ $ sudo apt install -y docker.io
```

and follow these [instructions](#custom-ops).

### Raspberry Pi

The instructions essentially follow those of Debian but mitigates issues that may arise:
- numpy may not work if installed via pip
- numpy installed via apt may not work in virtual environments
- TensorFlow is only currently available in version 1.11 for Raspberry Pi

The latter of these means that we have only tested using Raspberry Pi as servers.

First install Python 3:

```
./ $ sudo apt update
./ $ sudo apt upgrade -y
./ $ sudo apt install -y python3-pip git
```

followed by TensorFlow and its dependencies:

```
./ $ pip3 install tensorflow
```

However, the version of numpy installed this way may cause issues related to missing files. To get around these we can replace it with a version installed through apt:

```
./ $ pip3 uninstall -y numpy
./ $ sudo apt install -y python3-numpy
```

We finally close the repository:

```
./ $ git clone https://github.com/tf-encrypted/tf-encrypted.git
./ $ cd tf-encrypted
```

install TF Encrypted outside a virtual environment and without dependencies:

```
./tf-encrypted/ $ pip3 install -e . --no-deps
```

## IDE Setup

### VSCode

There are not specific requirements for using VSCode but the following
[`settings.json`](https://code.visualstudio.com/docs/python/settings-reference)
works well with our build process, where `<full Python path>` must be
replaced to match your system configuration.

```json
{
    "python.pythonPath": "<full Python path>",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.mypyEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.pydocstyleEnabled": true,
}
```
