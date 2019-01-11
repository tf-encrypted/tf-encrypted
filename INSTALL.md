# Installation

The most convinient way of installing is via the official [tf-encrypted](https://pypi.org/project/tf-encrypted/) pip package:

```
pip3 install tf-encrypted
```

However, for development or on certain platforms this is not sufficient. In these cases a source code installation is more fitting as described in this document.

Complete instructions for setting up on [macOS](#macos) and [Debian/Ubuntu](#debian-ubuntu) are given towards the end.

**Important:** For the rest of this document we assume that `pip` refers to version 3 of the tool, which will be the case if you are using a virtual environment such as [virtualenv](https://virtualenv.pypa.io/) or [conda](https://conda.io/). The steps for a [basic installation](#basics) can trivally be completed without this assumptions by simply using `pip3` instead. However, the [`Makefile`](./Makefile) used for [building custom ops](#custom-ops) is written with a virtual environment in mind and will hence require a bit of adaptation.

# Basics

All you need to get started is to have Python 3.5 or 3.6 installed along with [pip](https://pypi.org/project/pip/). The first step is then to clone the source code available on [GitHub](https://github.com/mortendahl/tf-encrypted):

```
./ $ git clone https://github.com/mortendahl/tf-encrypted.git
./ $ cd tf-encrypted
```

and install it as a pip package. The simplest way of doing this is by running

```
(venv) ./tf-encrypted/ $ pip install -e .
```

which will automatically install all needed dependencies, most notably [TensorFlow](https://www.tensorflow.org/install/) version 1.12 or above; if for some reason this is not desired then use

```
(venv) ./tf-encrypted/ $ pip install -e . --no-deps
```

instead and check that you have all dependencies installed if you encounter any problems later.

That's it - you should now have a working copy ready for development!

However, running code may at this point generate warnings related to sub-optimal performance and security. The reason for this is twofold: some features have not yet shipped as part of the official TensorFlow distribution, and some features rely on [custom ops](https://www.tensorflow.org/guide/extend/op) that first needs to be compiled. We address both below but stress that these steps can be skipped for initial experiments.

# Custom Ops

Certains operations, such as secure randomness generation, rely on C++ extentions of TensorFlow known as [custom ops](https://www.tensorflow.org/guide/extend/op). These come precompiled with the [tf-encrypted pip package](https://pypi.org/project/tf-encrypted/) but need to be manually compiled when installing from source code as we did above.

To do so you need the following system tools installed: curl, libtool, automake, and g++. These can typically be installed using your system's package manager (see below) and once done allows for building the custom ops using:

```
(venv) ./tf-encrypted/ $ make build
``` 

# Testing

To run unit tests as part of development you need to run 

```
(venv) ./tf-encrypted/ $ make test
``` 

after having installed flake8

```
(venv) ./tf-encrypted/ $ pip install flake8
```

# Custom TensorFlow

While tf-encrypted will work with the official release of [TensorFlow](https://pypi.org/project/tensorflow/) (version 1.12+), some features currently depend on improvements that have not yet been shipped. In particular, to get speed improvements by using int64 tensors instead of int100 tensors you currently need a custom build of TensorFlow.

We provide such builds for [macOS](https://storage.googleapis.com/dropoutlabs-tensorflow-builds/tensorflow-1.12.0-cp35-cp35m-macosx_10_7_x86_64.whl) and [Linux](https://storage.googleapis.com/dropoutlabs-tensorflow-builds/tensorflow-1.12.0-cp35-cp35m-linux_x86_64.whl) as a temporary solution until the next official release of TensorFlow is out (version 1.13), but no guarantees are made about them and they should be treated as pre-alpha.

Alternatively you can build TensorFlow on your own by e.g. following the [official instructions](https://www.tensorflow.org/install/source).

In both cases should you end up with a wheel file that you can install using pip:

```
pip install tensorflow-1.13.0-XXX.whl
```

tf-encrypted auto-detects which features are available so no further actions are needed.

# Complete Instructions

## macOS

<i>(coming)</i>

## Debian and Ubuntu

These steps have been tested on Debian 9, Ubuntu 16.04, and Ubuntu 18.04.

First install the system tools needed for basic installations:

```
$ sudo apt update
$ sudo apt upgrade -y
$ sudo apt install -y virtualenv python3-pip git
```

then create a virtual environment (on Ubuntu 16.04 we needed to `export LC_ALL=C` first):

```
./ $ virtualenv -p python3 venv
./ $ source venv/bin/activate
```

and follow the [basic instructions](#basics).

To compile custom ops first install the additional system tools

```
$ sudo apt install -y libtool automake g++ curl
```

and then follow [instructions](#custom-ops).
