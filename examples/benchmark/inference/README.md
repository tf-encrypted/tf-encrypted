# Convert TF Model to TFE Model and Make Private Prediction

This benchmark illustrates how TF Encrypted can be used to perform inference with very complicated neural network model, by converting a plain TF model into a private one in TFE.

## Running

Start the benchmark with the startup script `run-remote.sh` or `run-remote-4-cores.sh` (restrict each party to use 4 cores),
and you must explicitly specify which model to infer

```sh
./examples/benchmark/convert/run-remote.sh resnet50
```

You can also specify which tfe protocol to use and which remote config file to use

```sh
./examples/benchmark/convert/run-remote.sh resnet50 --protocol ABY3 --config config.json
```

By default, inference uses 64 bits for secret sharing, this gives enough precision in most cases.
We also give a option to use 128 bits for secret sharing by setting `--precision high`,
this will give you more precision, but at a cost of more computation time.

```sh
./examples/benchmark/aby3_profile/run-remote.sh resnet50 --precision high
```

