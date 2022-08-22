# Private Training for MNIST

This benchmark illustrates how TF Encrypted can be used to perform private training using several neural network models on the MNIST data set.

## Running

Make sure to have the training and test data sets downloaded before running the example:

```sh
python3 examples/benchmark/training/download.py
```

which will place the converted files in the `./data` subdirectory.

Then start the private training with the startup script `run-remote-network.sh` or `run-remote-network-4-cores.sh` (restrict each party to use 4 cores)

```sh
./examples/benchmark/training/run-remote-network.sh
```

You can play with 4 different network models (A, B, C, D) and 3 different optimizers (SGD, AMSgrad, Adam) by modifying corresponding lines in `private_network_training.py`.
