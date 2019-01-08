# Examples

This directory contains various examples of using tf-encrypted.

## Running

As an easy way of quickly getting started, all examples can be run locally by simply executing the corresponding entrypoint files. For instance, for [`???`]() one simply runs

```sh
python examples/???/run.py
```

This however runs everything on a single machine and hence doesn't provide any real security benefits. As such, each example can of course also be run in a remote setting where each party is running on a separate machine. Besides setting up and running TensorFlow servers on these machines the only thing to do is to define a suitable `config.json` file. Concretely,

```json
{
    'server0': '10.0.0.1',
    'server1': '10.0.0.2',
    'master':  '10.0.0.9',
 }
```

## GCP

To set up this on GCP