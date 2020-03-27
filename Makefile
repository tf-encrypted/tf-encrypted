# Global Variables used across many different rule types

# Definition of the default rule
all: test
.PHONY: all

# ###############################################
# Bootstrapping
#
# Rules for bootstrapping the Makefile such as checking for docker, python versions, etc.
# ###############################################
DOCKER_REQUIRED_VERSION=18.
PYTHON_REQUIRED_VERSION=3.6.
SHELL := /bin/bash

CURRENT_DIR=$(shell pwd)
PIP_PATH=$(shell which pip)
DOCKER_PATH=$(shell which docker)
CURRENT_TF_VERSION=$(shell python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null)

# Default platform
# PYPI doesn't allow linux build tags to be pushed and doesn't support
# specific operating systems such a ubuntu. It only allows build tags for linux
# to be pushed as manylinux.
DEFAULT_PLATFORM=manylinux1_x86_64

dockercheck:
ifeq (,$(DOCKER_PATH))
ifeq (,$(findstring $(DOCKER_REQUIRED_VERSION),$(shell docker version)))
ifeq (,$(BYPASS_DOCKER_CHECK))
	$(error "Docker version $(DOCKER_REQUIRED_VERSION) is required.")
endif
endif
endif

pythoncheck:
ifeq (,$(findstring $(PYTHON_REQUIRED_VERSION),$(shell python -V 2>&1)))
ifeq (,$(BYPASS_PYTHON_CHECK))
	$(error "Python version $(PYTHON_REQUIRED_VERSION) is required.")
endif
endif

pipcheck:
ifeq (,$(PIP_PATH))
ifeq (,$(BYPASS_PIP_CHECK))
	$(error "Pip must be installed")
endif
endif

bootstrap: pythoncheck pipcheck
	pip install -U pip setuptools
	pip install -r requirements.txt
	pip install -e .
	$(MAKE) build

# ###############################################
# Testing and Linting
#
# Rules for running our tests and for running various different linters
# ###############################################
test: pythoncheck
	pytest -n 8 -x -m "not slow and not convert_maxpool" tf_encrypted
	pytest -n 8 -x -m "slow" tf_encrypted
	pytest -n 8 -x -m "convert_maxpool" tf_encrypted

lint: pythoncheck
	flake8 tf_encrypted operations examples

fmt: pythoncheck
	isort --atomic --recursive tf_encrypted operations examples
	black tf_encrypted operations examples

typecheck: pythoncheck
	MYPYPATH=$(CURRENT_DIR):$(CURRENT_DIR)/stubs mypy tf_encrypted

.PHONY: lint fmt test typecheck

# ##############################################
# Documentation
#
# Rules for building our documentation
# ##############################################
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = docs/source
BUILDDIR      = build

SPHINX_BUILD_GOOGLE_DOCSTRINGS = sphinx-apidoc
SPHINX_NAPOLEAN_BUILD_DIR = docs/source/gen
SPHINX_PROJECT_DIR = tf_encrypted

CONVERT_DIR=tf_encrypted/convert
BUILD_RESERVED_SCOPES=$(CONVERT_DIR)/specops.yaml
$(BUILD_RESERVED_SCOPES): pythoncheck
	python -m tf_encrypted.convert.gen.generate_reserved_scopes

BUILD_CONVERTER_README=$(CONVERT_DIR)/gen/readme_template.md
$(BUILD_CONVERTER_README): $(BUILD_RESERVED_SCOPES) pythoncheck
	python -m tf_encrypted.convert.gen.generate_reserved_scopes

google-docstrings:
	@$(SPHINX_BUILD_GOOGLE_DOCSTRINGS) -fMeET "$(SPHINX_PROJECT_DIR)" -o "$(SPHINX_NAPOLEAN_BUILD_DIR)"

docs: $(BUILD_CONVERTER_README) google-docstrings
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: docs

# ###############################################
# Version Derivation
#
# Rules and variable definitions used to derive the current version of the
# source code. This information is also used for deriving the type of release
# to perform if `make push` is invoked.
# ###############################################
VERSION=$(shell [ -d .git ] && git describe --tags --abbrev=0 2> /dev/null | sed 's/^v//')
EXACT_TAG=$(shell [ -d .git ] && git describe --exact-match --tags HEAD 2> /dev/null | sed 's/^v//')
ifeq (,$(VERSION))
    VERSION=dev
endif
NOT_RC=$(shell git tag --points-at HEAD | grep -v -e -rc)

ifeq ($(EXACT_TAG),)
    PUSHTYPE=master
else
    ifeq ($(NOT_RC),)
	PUSHTYPE=release-candidate
    else
	PUSHTYPE=release
    endif
endif

releasecheck:
ifneq (yes,$(RELEASE_CONFIRM))
	$(error "Set RELEASE_CONFIRM=yes to really build and push release artifacts")
endif

.PHONY: releasecheck

# ###############################################
# Building Docker Image
#
# Builds a docker image for TF Encrypted that can be used to deploy and
# test.
# ###############################################
DOCKER_BUILD=docker build -t tfencrypted/tf-encrypted:$(1) -f Dockerfile $(2) .
docker: Dockerfile dockercheck
	$(call DOCKER_BUILD,latest,)

.PHONY: docker

# ###############################################
# Releasing Docker Images
#
# Using the docker build infrastructure, this section is responsible for
# authenticating to docker hub and pushing built docker containers up with the
# appropriate tags.
# ###############################################
DOCKER_TAG=docker tag tfencrypted/tf-encrypted:$(1) tfencrypted/tf-encrypted:$(2)
DOCKER_PUSH=docker push tfencrypted/tf-encrypted:$(1)

docker-logincheck:
ifeq (,$(DOCKER_USERNAME))
ifeq (,$(DOCKER_PASSWORD))
	$(error "Docker login DOCKER_USERNAME and DOCKER_PASSWORD environment variables missing")
endif
endif

docker-tag: dockercheck
	$(call DOCKER_TAG,latest,$(VERSION))

docker-push-tag: dockercheck
	$(call DOCKER_PUSH,$(VERSION))

docker-push-latest: dockercheck
	$(call DOCKER_PUSH,latest)

# Rely on DOCKER_USERNAME and DOCKER_PASSWORD being set inside CI or equivalent
# environment
docker-login: dockercheck docker-logincheck
	@echo "Attempting to log into docker hub"
	@docker login -u="$(DOCKER_USERNAME)" -p="$(DOCKER_PASSWORD)"

.PHONY: docker-login docker-push-lateset docker-push-tag docker-tag

# ###############################################
# Targets for pushing docker images
#
# The following are that are called dependent on the push type of the release.
# They define what actions occur depending no whether this is simply a build of
# master (or a branch), release candidate, or a full release.
# ###############################################

# For all builds on the master branch, build the container
docker-push-master: docker

# For all builds on the master branch, with an rc tag
docker-push-release-candidate: releasecheck docker-push-master docker-login docker-tag docker-push-tag

# For all builds on the master branch with a release tag
docker-push-release: docker-push-release-candidate docker-push-latest

# This command calls the right docker push rule based on the derived push type
docker-push: docker-push-$(PUSHTYPE)

.PHONY: docker-push docker-push-release docker-push-release-candidate docker-push-master

# ###############################################
# Targets for building pip packages for pypi
# ##############################################

pypi-version-check:
ifeq (,$(shell grep -e $(VERSION) setup.py))
	$(error "Version specified in setup.py does not match $(VERSION)")
endif
ifeq (,$(shell grep -e $(VERSION) meta.yaml))
	$(error "Version specified in meta.yaml does not match $(VERSION)")
endif
ifeq (,$(shell grep -e $(VERSION) docs/source/conf.py))
	$(error "Version specified in docs/source/conf.py does not match $(VERSION)")
endif

# default to manylinux
pypi-platform-check:
ifeq (,$(PYPI_PLATFORM))
PYPI_PLATFORM=$(DEFAULT_PLATFORM)
endif

pypi-build: pythoncheck pipcheck pypi-platform-check pypi-version-check build-all
	pip install --upgrade setuptools wheel twine
	rm -rf dist
ifeq ($(PYPI_PLATFORM),$(DEFAULT_PLATFORM))
	python setup.py sdist bdist_wheel --plat-name=$(PYPI_PLATFORM)
else
	python setup.py bdist_wheel --plat-name=$(PYPI_PLATFORM)
endif

.PHONY: pypi-build pypi-platform-check pypi-version-check

# ###############################################
# Targets for publishing to pypi
#
# These targets requires a PYPI_USERNAME, PYPI_PASSWORD, and PYPI_PLATFORM
# environment variables to be set to be executed properly.
# ##############################################

pypi-credentials-check:
ifeq (,$(PYPI_USERNAME))
ifeq (,$(PYPI_PASSWORD))
	$(error "Missing PYPI_USERNAME and PYPI_PASSWORD environment variables")
endif
endif

pypi-push-master: pypi-credentials-check pypi-build

pypi-push-release-candidate: releasecheck pypi-credentials-check pypi-build
	@echo "Attempting to upload to pypi"
	twine upload -u="$(PYPI_USERNAME)" -p="$(PYPI_PASSWORD)" dist/*

pypi-push-release: pypi-push-release-candidate

pypi-push: pypi-push-$(PUSHTYPE)

.PHONY: pypi-push pypi-push-release pypi-push-release-candidate pypi-push-master pypi-credentials-check

# ###############################################
# Pushing Artifacts for a Release
#
# The following are meta-rules for building and pushing various different
# release artifacts to their intended destinations.
# ###############################################

push:
	@echo "Attempting to build and push $(VERSION) with push type $(PUSHTYPE) - $(EXACT_TAG)"
	# make docker-push
	make pypi-push
	@echo "Done building and pushing artifacts for $(VERSION)"

.PHONY: push

# ###############################################
# libsodium
# ###############################################

LIBSODIUM_VER_TAG=1.0.17
LIBSODIUM_DIR=build/libsodium-$(LIBSODIUM_VER_TAG)
LIBSODIUM_INSTALL = $(shell pwd)/build
LIBSODIUM_OUT = $(LIBSODIUM_INSTALL)/lib/libsodium.a

$(LIBSODIUM_OUT):
	curl -OL https://github.com/jedisct1/libsodium/archive/$(LIBSODIUM_VER_TAG).tar.gz
	mkdir -p build
	tar -xvf $(LIBSODIUM_VER_TAG).tar.gz -C build
	cd $(LIBSODIUM_DIR) && ./autogen.sh && ./configure --disable-shared --enable-static \
		--disable-debug --disable-dependency-tracking --with-pic --prefix=$(LIBSODIUM_INSTALL)
	$(MAKE) -C $(LIBSODIUM_DIR)
	$(MAKE) -C $(LIBSODIUM_DIR) install

# ###############################################
# Common TensorFlow flags for custom ops
# ###############################################

TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))' 2>/dev/null)
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))' 2>/dev/null)

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	FINAL_TF_LFLAGS = $(TF_LFLAGS)
endif
ifeq ($(UNAME_S),Darwin)
	FINAL_TF_LFLAGS = $(word 1,$(TF_LFLAGS)) -ltensorflow_framework
endif

# ###############################################
# Secure Random Shared Object
#
# Rules for the shared object for secure random.
# ###############################################

SECURE_OUT_PRE = tf_encrypted/operations/secure_random/secure_random_module_tf_
SECURE_IN = operations/secure_random/secure_random.cc
SECURE_IN_H = operations/secure_random/generators.h

$(SECURE_OUT_PRE)$(CURRENT_TF_VERSION).so: $(LIBSODIUM_OUT) $(SECURE_IN) $(SECURE_IN_H)
	mkdir -p tf_encrypted/operations/secure_random

	g++ -std=c++11 -shared $(SECURE_IN) -o $(SECURE_OUT_PRE)$(CURRENT_TF_VERSION).so \
		-fPIC $(TF_CFLAGS) $(FINAL_TF_LFLAGS) -O2 -I$(LIBSODIUM_INSTALL)/include -L$(LIBSODIUM_INSTALL)/lib -lsodium

# ###############################################
# Sodium Primitives Shared Object
#
# Rules for the shared object for sodium primitives.
# ###############################################

SODIUM_OPS_PRE = tf_encrypted/primitives/sodium/sodium_module_tf_
SODIUM_OPS_IN = operations/sodium/ops.cc operations/sodium/easy_box_kernels.cc

$(SODIUM_OPS_PRE)$(CURRENT_TF_VERSION).so: $(LIBSODIUM_OUT) $(SODIUM_OPS_IN)
	g++ -std=c++11 -shared $(SODIUM_OPS_IN) -o $(SODIUM_OPS_PRE)$(CURRENT_TF_VERSION).so \
		-fPIC $(TF_CFLAGS) $(FINAL_TF_LFLAGS) -O2 -I$(LIBSODIUM_INSTALL)/include -L$(LIBSODIUM_INSTALL)/lib -lsodium


# ###############################################
# Build
# ###############################################

build: $(SECURE_OUT_PRE)$(CURRENT_TF_VERSION).so $(SODIUM_OPS_PRE)$(CURRENT_TF_VERSION).so

build-all:
	pip install tensorflow==1.15.2
	$(MAKE) $(SECURE_OUT_PRE)1.15.2.so

.PHONY: build build-all

clean:
	$(MAKE) -C $(LIBSODIUM_DIR) uninstall
	rm -fR build
	rm -f $(LIBSODIUM_VER_TAG).tar.gz
	find ./tf_encrypted/operations -name '*.so' -delete

clean-pycache:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

.PHONY: clean clean-pycache
