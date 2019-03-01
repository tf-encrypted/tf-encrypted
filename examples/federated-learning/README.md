# Secure Aggregation for Federated Learning

This example shows how TF Encrypted can be used to perform secure aggregation for federated learning, where a *model owner* is training a model by repeatedly asking a set of *data owners* to compute gradients on their locally held data set. As a way of reducing the privacy leakage, only the mean gradient is revealed to the model owner in each iteration.

<p align="center"><img src="./flow.png" style="width: 50%;"/></p>

## Computation

The above players are represented via two classes, `ModelOwner` and `DataOwner`, and linked together as follows:

```python
# instantiate the model owner and the model it wishes to train
model_owner = ModelOwner('model-owner')

# instantiate data owners and build copy of the model on their devices linked to the model owner's
data_owners = [
    DataOwner('data-owner-0', model_owner.build_training_model),
    DataOwner('data-owner-1', model_owner.build_training_model),
    DataOwner('data-owner-2', model_owner.build_training_model),
]
```

Then, the result of `compute_gradient` of each data owner is used as a private input into a secure computation, in this case between three *compute servers* as needed by the default Pond protocol.

```python
# collect encrypted gradients from data owners
model_grads = zip(*(
    tfe.define_private_input(data_owner.player_name, data_owner.compute_gradient)
    for data_owner in data_owners
))

# compute mean of gradients (without decrypting)
aggregated_model_grads = [
    tfe.add_n(grads) / len(grads)
    for grads in model_grads
]

# send the encrypted aggregated gradients to the model owner for it to decrypt and update
iteration_op = tfe.define_output(
    model_owner.player_name,
    aggregated_model_grads,
    model_owner.update_model
)
```

Finally, we simply run the update procedure for a certain number of iterations:

```python
with tfe.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(model_owner.ITERATIONS):
        sess.run(iteration_op)
```

## Running

Make sure to have the training and test data sets downloaded before running the example:

```sh
python3 examples/federated-learning/download.py
```

which will place the converted files in the `./data` subdirectory

To then run locally use:

```sh
python3 examples/federated-learning/run.py
```

or remotely using:

```sh
python3 examples/federated-learning/run.py config.json
```

See more details in the [documentation](/docs/RUNNING.md).
