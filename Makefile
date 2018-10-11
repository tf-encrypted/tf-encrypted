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
PYTHON_REQUIRED_VERSION=3.5.
TENSORFLOW_REQUIRED_VERSION=1.11
SHELL := /bin/bash

CURRENT_DIR=$(shell pwd)
PIP_PATH=$(shell which pip)
DOCKER_PATH=$(shell which docker)
CURRENT_TF_VERSION=$(shell python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null)

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

tensorflowcheck:
ifeq (,$(findstring $(TENSORFLOW_REQUIRED_VERSION),$(CURRENT_TF_VERSION)))
ifeq (,$(BYPASS_TENSORFLOW_CHECK))
	$(error "Tensorflow version $(TENSORFLOW_REQUIRED_VERSION) is required.")
endif
endif

bootstrap: pythoncheck pipcheck
	pip install -r requirements.txt
	pip install -e .

# ###############################################
# Testing and Linting
#
# Rules for running our tests and for running various different linters
# ###############################################
test: lint pythoncheck tensorflowcheck
	python examples/convert.py
	python examples/inputs.py
	python examples/int32.py
	python examples/int100.py
	python examples/matmul.py
	python examples/pond-simple.py
	python examples/securenn-playground.py
	python examples/federated-average/run.py
	python -m unittest discover

lint: pythoncheck tensorflowcheck
	flake8

typecheck: pythoncheck tensorflowcheck
	MYPYPATH=$(CURRENT_DIR):$(CURRENT_DIR)/stubs mypy tensorflow_encrypted


.PHONY: lint test typecheck

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
# Builds a docker image for tf-encrypted that can be used to deploy and
# test.
# ###############################################
docker: Dockerfile dockercheck
	docker build -t mortendahl/tf-encrypted:latest -f Dockerfile .

.PHONY: docker

# ###############################################
# Releasing Docker Images
#
# Using the docker build infrastructure, this section is responsible for
# authenticating to docker hub and pushing built docker containers up with the
# appropriate tags.
# ###############################################
DOCKER_TAG=docker tag mortendahl/tf-encrypted:latest mortendahl/tf-encrypted:$(1)
DOCKER_PUSH=docker push mortendahl/tf-encrypted:$(1)

docker-logincheck:
ifeq (,$(DOCKER_USERNAME))
ifeq (,$(DOCKER_PASSWORD))
	$(error "Docker login DOCKER_USERNAME and DOCKER_PASSWORD environment variables missing")
endif
endif

docker-tag: dockercheck
	$(call DOCKER_TAG,$(VERSION))

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

# For all builds onthe master branch, with an rc tag
docker-push-release-candidate: releasecheck docker-push-master docker-login docker-tag docker-push-tag

# For all builds on the master branch with a release tag
docker-push-release: docker-push-release-candidate docker-push-latest

# This command calls the right docker push rule based on the derived push type
docker-push: docker-push-$(PUSHTYPE)

.PHONY: docker-push docker-push-release docker-push-release-candidate docker-push-master

# ###############################################
# Targets for publishing to pypi
#
# These targets required a PYPI_USERNAME and PYPI_PASSWORD environment
# variables to be set to be executed properly.
# ##############################################

pypicheck: pipcheck pythoncheck tensorflowcheck
ifeq (,$(PYPI_USERNAME))
ifeq (,$(PYPI_PASSWORD))
	$(error "Missing PYPI_USERNAME and PYPI_PASSWORD environment variables")
endif
endif

pypi-version-check:
ifeq (,$(shell grep -e $(VERSION) setup.py))
	$(error "Version specified in setup.py does not match $(VERSION)")
endif

pypi-push-master: pypicheck pypi-version-check
	pip install --user --upgrade setuptools wheel twine
	rm -rf dist
	python setup.py sdist bdist_wheel

pypi-push-release-candidate: releasecheck pypi-push-master
	@echo "Attempting to upload to pypi"
	@PATH=\$PATH:~/.local/bin twine upload -u="$(PYPI_USERNAME)" -p="$(PYPI_PASSWORD)" dist/*

pypi-push-release: pypi-push-release-candidate

pypi-push: pypi-push-$(PUSHTYPE)

.PHONY: pypi-push-master pypi-push-release-candidate pypi-push-release pypi-push pypicheck pypi-version-check

# ###############################################
# Pushing Artifacts for a Release
#
# The following are meta-rules for building and pushing various different
# release artifacts to their intended destinations.
# ###############################################
push:
	@echo "Attempting to build and push $(VERSION) with push type $(PUSHTYPE) - $(EXACT_TAG)"
	make docker-push
	make pypi-push
	@echo "Done building and pushing artifacts for $(VERSION)"

.PHONY: push
