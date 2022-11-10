# Joint Private Training

This benchmark illustrates how TF Encrypted can be used to perform joint private training with horizontally or vertically splited dataset.

## Running

Start the joint private training with the startup script `run-(horizontal/vertical)-remote-network.sh`, 
and you must explicitly specify which model to train and which dataset to use

```sh
./examples/benchmark/mnist/run-(horizontal/vertical)-remote-network.sh network_a Mnist
```

You can also specify how many epochs to run, which tfe protocol to use and which remote config file to use

```sh
./examples/benchmark/mnist/run-(horizontal/vertical)-remote-network.sh network_a Mnist --epochs 10 --protocol ABY3 --config config.json
```