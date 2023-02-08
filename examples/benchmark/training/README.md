# Private Training

This benchmark illustrates how TF Encrypted can be used to perform private training using several neural network models and datasets.

## Running

Start the private training with the startup script `run-remote.sh` or `run-remote-4-cores.sh` (restrict each party to use 4 cores), 
and you must explicitly specify which model to train and which dataset to use.


```sh
./examples/benchmark/mnist/run-remote.sh network_a Mnist
```

You can also specify how many epochs to run, which tfe protocol to use and which remote config file to use

```sh
./examples/benchmark/mnist/run-remote.sh network_a Mnist --epochs 10 --protocol ABY3 --config config.json
```

By default, training uses 64 bits for secret sharing, this gives enough precision in most cases.
We also give a option to use 128 bits for secret sharing by setting `--precision high`,
this will give you more precision, but at a cost of more computation time.

```sh
./examples/benchmark/aby3_profile/run-remote.sh network_a Mnist --precision high
```

You can play with 4 different models:
- [`network_a`](../../models/network_a.py) 
- [`network_b`](../../models/network_b.py) 
- [`network_c`](../../models/network_c.py)
- [`network_d`](../../models/network_d.py)
- [`logistic_regression`](../../models/logistic_regression.py)

and three different datasets:
- [`Mnist`](../../../tf_encrypted/keras/datasets/mnist.py) Handwritten digits
- [`LogisticArtificial`](../../../tf_encrypted/keras/datasets/logistic_artificial.py) Artificial dataset for logistic regression
- [`LRMnist`](./lr_mnist_dataset.py) Mnist dataset classified into two classes: small digits (0-4) vs large digits (5-9)

`network_a`, `network_b`, `network_c`, `network_d` can be trained on `Mnist` dataset,
`logistic_regression` can be trained on `LogisticArtificial` and `LRMnist` dataset. 

You could also play with different optimizers (SGD, AMSgrad, Adam) by modifying corresponding lines in private_training.py.