.. tf-encrypted documentation master file, created by
   sphinx-quickstart on Tue Oct  9 10:51:37 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

tf-encrypted
========================================

tf-encrypted is an open source Python library built on top of `TensorFlow`_ focused on making it easy for researchers and practitioners to experiment with privacy-preserving machine learning without needing to be an expert in machine learning, cryptography, distributed systems, or high performance computing.

More detailed information on the architecture, performance, and security considerations of this project are contained in `Private Machine Learning in TensorFlow using Secure Computation`_ paper which introduced this project.

.. _TensorFlow: https://tensorflow.org
.. _Private Machine Learning in TensorFlow using Secure Computation: https://linkhere.org

- **Simple**: It should be easy to get started, experiment with, use, benchmark, and deeply integrate with pre-existing data science workflows.
- **Extensible**: Extending tf-encrypted to experiment with new protocols, tensor implementations, and machine learning algorithms should be a first class citizen.
- **Performant**: All of the fastest known implementations of protocols, tensor implementations, and machine learning algorithms in a private setting should exist within this library.
- **Secure**: A user should have to go out of their way to use the library in a non-secure, non-private fashion. New protocols are marked as experimental until they've met our high bar of security.
- **Community-Driven**: Community-first as a decision making framework. We can only achieve these goals by building a community of researchers, contributors, and users around the project.

Checkout the `Getting Started`_ guide to learn how to get up and running with private machine learning.

You can view the project source, contribute, and asks questions on `GitHub`_.

.. _Getting Started: usage/getting_started.html
.. _GitHub: https://github.com/mortendahl/tf-encrypted


-----------------------
License
-----------------------

This project is licensed under the Apache License, Version 2.0 (see `License`_). Copyright as specified in the `NOTICE`_ contained in the code base.

.. _License: https://github.com/mortendahl/tf-encrypted/blob/master/LICENSE
.. _NOTICE: https://github.com/mortendahl/tf-encrypted/blob/master/NOTICE

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   usage/installation
   usage/getting_started

.. toctree::
   :maxdepth: 5
   :caption: API

   api/protocol/index
   api/session
   api/config/config
   api/layers/index
