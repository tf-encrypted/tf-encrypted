# Profiling Operations in the ABY3 Protocol 

This benchmark profiles some complex operations (e.g., sort) in the ABY3 protocol.

## Running

Start the benchmark with the startup script `run-remote.sh` or `run-remote-4-cores.sh` (restrict each party to use 4 cores)

```sh
./examples/benchmark/aby3_profile/run-remote.sh test_sort_performance
```
You can replace `test_sort_performance` with other benchmark functions located in `aby3_profile.py`.
