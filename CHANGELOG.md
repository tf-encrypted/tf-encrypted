# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0]

**Added**

- SecureNN with int64 support has landed.
- Cryptographically secure random numbers feature has been implemented but not integrated.
- Three new ops have been added to Pond and the converter: Pad, BatchToSpaceND, SpaceToBatchND

**Changed**

- There are now separate wheels published to pypi for MacOS and linux.
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
