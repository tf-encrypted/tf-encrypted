# Private Training

This benchmark illustrates how TF Encrypted can be used to perform private training using several neural network models and datasets.

## Running

Start the private training with the startup script `run-remote-network.sh` or `run-remote-network-4-cores.sh` (restrict each party to use 4 cores), 
and you must explicitly specify which model to train and which dataset to use

```sh
./examples/benchmark/mnist/run-remote-network.sh network_a Mnist
```

You can also specify how many epochs to run, which tfe protocol to use and which remote config file to use

```sh
./examples/benchmark/mnist/run-remote-network.sh network_a Mnist --epochs 10 --protocol ABY3 --config config.json
```