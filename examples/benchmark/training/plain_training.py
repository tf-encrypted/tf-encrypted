# pylint:  disable=redefined-outer-name
"""An example of performing plain training with various model and dataset.
"""
import argparse
import sys

import tensorflow as tf

from tf_encrypted.keras.datasets import *  # noqa:F403,F401

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a plain TF model")
    parser.add_argument(
        "model_name",
        metavar="MODEL NAME",
        type=str,
        help="name of model to be trained",
    )
    parser.add_argument(
        "data_name",
        metavar="DATASET NAME",
        type=str,
        help="name of dataset which model trained on",
    )
    parser.add_argument(
        "--epochs",
        metavar="EPOCHS",
        type=int,
        default=5,
        help="how many epochs to run",
    )
    args = parser.parse_args()

    # import all models
    sys.path.append("../../")
    from models import *  # noqa:F403,F401

    # set train and test data
    Dataset = globals()[args.data_name + "Dataset"]
    train_dataset = Dataset(batch_size=128)
    train_data_iter = train_dataset.generator_builder()()
    test_dataset = Dataset(batch_size=100, train=False)
    test_data_iter = test_dataset.generator_builder()()

    # set model to be trained
    model = globals()[args.model_name](
        train_dataset.batch_shape, train_dataset.num_classes, private=False
    )
    if train_dataset.num_classes > 1:
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        metrics = ["categorical_accuracy"]
    else:
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metrics = ["binary_accuracy"]
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    # optimizer = tf.keras.optimizers.AMSgrad(learning_rate=0.001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer, loss, metrics=metrics)

    print("Train model")
    model.fit(
        x=train_data_iter, epochs=args.epochs, steps_per_epoch=train_dataset.iterations
    )

    print("Set trained weights")
    model_2 = globals()[args.model_name](
        test_dataset.batch_shape, test_dataset.num_classes, private=False
    )
    model_2.set_weights(model.weights)
    model_2.compile(optimizer, loss, metrics=metrics)

    print("Evaluate")
    result = model_2.evaluate(x=test_data_iter, steps=None)

    print(result)
