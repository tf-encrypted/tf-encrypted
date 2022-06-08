# Convert TF Graph to TFE Graph and Make Private Prediction

This benchmark illustrates how TF Encrypted can be used to perform inference with a very complicated neural network model (Resnet50), by converting a plain TF model into a private one in TFE.

## Running

Start the benchmark with the startup script `run-remote.sh` or `run-remote-4-cores.sh` (restrict each party to use 4 cores)

```sh
./examples/benchmark/convert/run-remote.sh
```
