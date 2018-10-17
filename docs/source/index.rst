.. tf-encrypted documentation master file, created by
   sphinx-quickstart on Tue Oct  9 10:51:37 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

tf-encrypted
========================================

.. image:: https://img.shields.io/badge/status-alpha-blue.svg
.. image:: https://img.shields.io/github/license/mortendahl/tf-encrypted.svg
.. image:: https://img.shields.io/pypi/v/tf-encrypted.svg
.. image:: https://circleci.com/gh/mortendahl/tf-encrypted/tree/master.svg?style=svg

This library provides a layer on top of TensorFlow for doing machine learning on encrypted data as initially described in `Secure Computations as Dataflow Programs`_, with the aim of making it easy for researchers and practitioners to experiment with private machine learning using familiar tools and without being an expert in both machine learning and cryptography. To this end the code is structured into roughly three modules:

.. _Secure Computations as Dataflow Programs: https://mortendahl.github.io/2018/03/01/secure-computation-as-dataflow-programs/

- secure operations for computing on encrypted tensors
- basic machine learning operations built on top of these
- ready-to-use components for private prediction and training

that are all exposed through Python interfaces, and all resulting in ordinary TensorFlow graphs for easy integration with other TensorFlow mechanisms and efficient execution.

Several contributors have put resources into the development of this library, most notably `Dropout Labs`_ and members of the `OpenMined`_ community.

.. _Dropout Labs: https://dropoutlabs.com/
.. _OpenMined: https://www.openmined.org/

**Important**: this is experimental software that should not be used in production for security reasons.


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
