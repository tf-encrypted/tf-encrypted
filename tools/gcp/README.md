# Experimenting on Google Cloud Platform

A few tools for making it slightly easier to run experiments on the Google Cloud Platform.

All steps here assume that you have a Google Cloud account and that you can install Google Cloud SDK. Please see [this](https://cloud.google.com/sdk/install) link for details on how install the SDK for all platforms.

If not already done you'll need to clone the git repository `https://github.com/tf-encrypted/tf-encrypted`. These commands will be run from within the repo (`cd tf-encrypted/tools/gcp`).

## Running

First, we need to specify all of the instance names we need. For example, for running a private prediction we only need three `server0`, `server1` and `server2`. We export an environement variable called `INSTANCE_NAMES` containing a list of the names.

```shell
export INSTANCE_NAMES="server0 server1 server2"
```

### Setup instances

Next we can launch the instances with a helper script:

```shell
./create $INSTANCE_NAMES
```

To use another image other than the default `docker.io/tfencrypted/tf-encrypted:latest` image you use an environment variable to specify which one

```shell
TFE_IMAGE=docker.io/tfencrypted/tf-encrypted:0.5.8 ./create $INSTANCE_NAMES
```

NOTE: docker.io must be passed as the gcloud command expect the full path.

Alternatively, if they have already been created but are currently terminated, simply start them again with

```shell
./start $INSTANCE_NAMES
```

These commands causes the instances to launch a docker container that runs the TF Encrypted server. At this point they are waiting for a configuration to use to connect to other servers.

### Linking Instances

We can generate and share a configuration file amongst all of the instance with:

```shell
./link $INSTANCE_NAMES
```

This uses the instances external addresses to connect one another and also opens a port on each instance at 4440.

### Cleaning Up

Once done, the instances can either simply be stopped with:

```shell
./stop $INSTANCE_NAMES
```

or destroyed entirely with

```shell
./delete $INSTANCE_NAMES
```