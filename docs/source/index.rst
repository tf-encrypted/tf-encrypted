.. tf-encrypted documentation master file, created by
   sphinx-quickstart on Tue Oct  9 10:51:37 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TF Encrypted API Docs
========================================

TF Encrypted is a framework for encrypted machine learning in `TensorFlow`_. It looks and feels like TensorFlow, taking advantage of the ease-of-use of the Keras API while enabling training and prediction over encrypted data. Under the hood, TF Encrypted integrates state-of-the-art cryptography like `secure multi-party computation`_, and `homomorphic encryption`_. TF Encrypted aims to make privacy-preserving machine learning readily available, without requiring expertise in cryptography, distributed systems, or high performance computing.

TF Encrypted focuses on:

- **Usability**: The API and its underlying design philosophy make it easy to get started, use, and integrate privacy-preserving technology into pre-existing machine learning processes.
- **Extensibility**: The architecture supports and encourages experimentation and benchmarking of new cryptographic protocols and machine learning algorithms.
- **Performance**: Optimizing for tensor-based applications and relying on TensorFlow's backend means runtime performance comparable to that of specialized stand-alone frameworks.
- **Community**: With a primary goal of pushing the technology forward the project encourages collaboration and open source over proprietary and closed solutions.
- **Security**: Cryptographic protocols are evaluated against strong notions of security and known limitations are highlighted.

This page only contains API documentation. Checkout the `examples`_ on github to learn how to get up and running with private machine learning.

You can view the project source, contribute, and asks questions on `GitHub`_.

.. _TensorFlow: https://www.tensorflow.org
.. _secure multi-party computation: https://en.wikipedia.org/wiki/Secure_multi-party_computation
.. _homomorphic encryption: https://en.wikipedia.org/wiki/Homomorphic_encryption
.. _examples: https://github.com/tf-encrypted/tf-encrypted/tree/master/examples
.. _GitHub: https://github.com/tf-encrypted/tf-encrypted


-----------------------
License
-----------------------

This project is licensed under the Apache License, Version 2.0 (see `License`_). Copyright as specified in the `NOTICE`_ contained in the code base.

.. _License: https://github.com/mortendahl/tf-encrypted/blob/master/LICENSE
.. _NOTICE: https://github.com/mortendahl/tf-encrypted/blob/master/NOTICE

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: API

   gen/tf_encrypted.keras
