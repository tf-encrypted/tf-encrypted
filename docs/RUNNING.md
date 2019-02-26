# Running

This document describes how to run secure computations using TF Encrypted. There are two ways of doing so as detailed below:

- Using a `LocalConfig` object: this simulates the players on the local machine using different threads, and as such is completely insecure; however, it is very convenient for testing purposes

- Using a `RemoteConfig` object: this orchestrates a distributed computation between ordinary TensorFlow servers, typically running on distinct machines in a network; this is secure and represents how computations should actually be run

New configurations are specified using `tfe.set_config()`. The default is `LocalConfig()`.

See also the [installation instructions](/docs/INSTALL.md) for more information on installing TF Encrypted, and [our GCP documentation](/tools/gcp/) for more instructions using it on the Google Cloud Platform.

## Simulated

As noted earlier, `LocalConfig` which serves as a convenient way of simulating computations by running everything on the local machine, using different threads for each player so that TensorBoard may still be used inspect which players performed which operations.

By default this configuration will create new players as needed and will hence typical be specified simply by `tfe.set_config(LocalConfig())`. The complete set of players will be logged during each execution, for instance:

```sh
INFO:tf_encrypted:Players: ['server0', 'server1', 'server2', 'input-provider', 'result-receiver']
```

which is a useful way of verifying who is involved in a given execution and may serve as a starting point for defining hostmap configurations as used by networked configurations. This behaviour can be changed by instead specified a fixed set of players: `tfe.set_config(LocalConfig(player_names, auto_add_unknown_players=False))`.

Note that this way of running computations does not provide any security. Moreover, it may not give accurate performance numbers and may even be slower than running in a networked setting.

## Networked

The overall procedure for running computations between distinct machines in a network is basically as follows:

1. Specify and distribute a `config.json` hostmap file
2. Launch servers on each machine using `python3 -m tf_encrypted.player --config config.json`
3. Run scripts using `tfe.set_config(tfe.RemoteConfig.load('config.json'))`

As a concrete example, say we want to run a computation involving three servers, an `input-provider`, and a `result-receiver`. We then start by creating a `config.json` file reflecting our network setup:

```json
{
    "server0": "10.0.0.10:4440",
    "server1": "10.0.0.11:4440",
    "server2": "10.0.0.12:4440",
    "input-provider": "10.0.0.20:4440",
    "result-receiver": "10.0.0.30:4440"
}
```

Note that all machines in the hostmap must be able to talk to each other; in particular, that proper firewall settings are in place for the specified ports.

Having distributed this file to every machine we next launch a TensorFlow server on each. Assuming that TF Encrypted has already been installed, the easiest way of during this is to simply use `python3 -m tf_encrypted.player`, which will parse the configuration file and start an ordinary TensorFlow server on the corresponding endpoint. For our concrete example we have:

```sh
user@10.0.0.10 $ python3 -m tf_encrypted.player server0 --config config.json
user@10.0.0.11 $ python3 -m tf_encrypted.player server1 --config config.json
user@10.0.0.12 $ python3 -m tf_encrypted.player server2 --config config.json
user@10.0.0.20 $ python3 -m tf_encrypted.player input-provider --config config.json
user@10.0.0.30 $ python3 -m tf_encrypted.player result-receiver --config config.json
```

Finally, assuming our computation is defined by some Python script we simply add the following to the top of it:

```python
config = tfe.RemoteConfig.load('config.json')
tfe.set_config(config)
```

Running the script may be done from anywhere, including e.g. a distinct laptop, with the only requirement that it has network access to the *first host* listed in `config.json` (in our case `server0`) which will be the machine responsible for coordinating the computation. Since this involves some amount of compute it may be convenient to set aside a dedicated `master` machine, listed first in the configuration:

```json
{
    "master": "10.0.0.1:4440",
    "server0": "10.0.0.10:4440",
    "server1": "10.0.0.11:4440",
    "server2": "10.0.0.12:4440",
    "input-provider": "10.0.0.20:4440",
    "result-receiver": "10.0.0.30:4440"
}
```

Note that maybe of the [examples](/examples/) can be run by simply passing in a path to the configuration file, e.g.:

```sh
python3 examples/simple-average/run.py config.json
```

## Inspection and Tracing

When using `tfe.Session` and specifying a tag, e.g. `sess.run(fetch, tag='average')`, summary files for TensorBoard may be written by setting `TFE_STATS=1` in the environment, say:

```sh
TFE_STATS=1 python3 ./examples/simple-average/run.py
```

Tracing files may additionally be written by also setting `TFE_TRACE=1`.
