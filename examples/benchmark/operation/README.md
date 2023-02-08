# Profiling Operations in TF Encrypted 

This benchmark profiles some complex operations (e.g., sort) in tfe.

## Running

Start the benchmark with the startup script `run-remote.sh` or `run-remote-4-cores.sh` (restrict each party to use 4 cores),
and you must explicitly specify which test to run

```sh
./examples/benchmark/aby3_profile/run-remote.sh test_sort_performance
```

You can also specify which tfe protocol to use and which remote config file to use

```sh
./examples/benchmark/aby3_profile/run-remote.sh test_sort_performance --protocol ABY3 --config config.json
```

By default, test uses 64 bits for secret sharing, this gives enough precision in most cases.
We also give a option to use 128 bits for secret sharing by setting `--precision high`,
this will give you more precision, but at a cost of more computation time.

```sh
./examples/benchmark/aby3_profile/run-remote.sh test_sort_performance --precision high
```