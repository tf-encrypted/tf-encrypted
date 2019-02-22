# Secure Aggregation for Federated Learning

This example shows how TF Encrypted can be used to perform secure aggregation for federated learning, where a *model owner* is training a model by repeatedly asking a set of *data owners* to compute gradients on their locally held data set. As a way of reducing the privacy leakage, only the mean gradient is revealed to the model owner in each iteration.

<p align="center"><img src="./flow.png" style="width: 50%;"/></p>

## Computation

The above players are represented via two classes, `ModelOwner` and `DataOwner`, and linked together as follows:

```python
model_owner = ModelOwner('model-owner')
data_owners = [
    DataOwner('data-owner-0', model_owner.build_training_model),
    DataOwner('data-owner-1', model_owner.build_training_model),
    DataOwner('data-owner-2', model_owner.build_training_model),
]
```

Then, the result of `compute_gradient` of each data owner is used as a private input into a secure computation, in this case between three *compute servers* as needed by the default Pond protocol.

```python
model_grads = zip(*(
    tfe.define_private_input(data_owner.player_name, data_owner.compute_gradient)
    for data_owner in data_owners
))

with tf.name_scope('secure_aggregation'):
    aggregated_model_grads = [
        tfe.add_n(grads) / len(grads)
        for grads in model_grads
    ]
```

Finally, the aggregated gradients are sent to the model owner for it to update its model:

```python
iteration_op = tfe.define_output(model_owner.player_name, aggregated_model_grads, model_owner.update_model)
```

## Running

Make sure to have downloaded the training and test data sets before running the actual example:

```sh
python examples/federated-learning/download.py
```

### Locally

The example may be run locally using a `LocalConfig`:

```sh
python examples/federated-learning/run.py
```

which also reveals the full set of players involved:

```
INFO:tf_encrypted:Players: ['server0', 'server1', 'server2', 'model-owner', 'data-owner-0', 'data-owner-1', 'data-owner-2']
```

### Remotely

By specifying a configuration file the example may also be run remotely on distinct machines:

```sh
python examples/federated-learning/run.py config.json
```
