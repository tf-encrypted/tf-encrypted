"""Plaintext benchmark for"""
import argparse
import os
import time

import tensorflow as tf
import tf2onnx
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python import errors_impl as errors
from utils import data_prep
from utils import read_one_row

parser = argparse.ArgumentParser()
parser.add_argument(
    "--predict",
    type=int,
    default="-1",
    help="Do a prediction on specified row of input file",
)
parser.add_argument(
    "--input_file",
    type=str,
    default="final_data_with_feature_engineered.csv",
    help=("Location of input, defaults to ", "final_data_with_feature_engineered.csv"),
)
parser.add_argument(
    "--bench_prediction",
    type=int,
    default=-1,
    help=("Benchmark prediction by doing n iterations", "and taking the average"),
)
config = parser.parse_args()

epochs = 20
batch_size = 256

checkpoint_path = "./saved_models/train"

predict_row = config.predict
input_file = config.input_file
bench_prediction = config.bench_prediction


def export_to_onnx(input_shape, model, file_path):
    input_spec = []
    input_spec.append(tf.TensorSpec(input_shape, tf.float32, name="input"))
    model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=input_spec, output_path=file_path
    )
    print("saved the onnx graph (ready for inference) at: ", file_path)


def print_nodes(graph):
    """Print a list of nodes from a onnx graph."""
    print([n.name for n in graph.node])


def build_model(input_shape):
    """Build a logistic regression model with tf.keras."""
    model = keras.Sequential(
        [
            layers.Dense(
                1, use_bias=False, activation="sigmoid", input_shape=[input_shape]
            ),
        ]
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.train.AdamOptimizer(),
        metrics=["accuracy"],
    )

    return model


def train(train_x_df, train_y_df):
    """Train a logistic regressor on the dataset."""
    x = list(train_x_df.columns.values)
    model = build_model(len(x))

    os.makedirs("./saved_models", exist_ok=True)

    cp_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_weights_only=True, save_best_only=True, verbose=1
    )

    # first 80 percent for training
    train_x = train_x_df[1:246005]
    train_y = train_y_df[1:246005]

    # other 20 percent for evaluating
    eval_x = train_x_df[246006 : len(train_x_df) - 1]
    eval_y = train_y_df[246006 : len(train_y_df) - 1]

    # train model
    model.fit(
        train_x,
        train_y,
        epochs=epochs,
        validation_split=0.2,
        verbose=0,
        batch_size=batch_size,
        callbacks=[cp_callback],
    )

    print("done training")

    # export the tensorflow model to a onnx model file
    # for loading in tfe and secure enclave
    export_to_onnx(
        x,
        model,
        "./house_credit_default.onnx",
    )

    # evaluate the model using AUC, the metric used in the kaggle competition
    loss = model.evaluate(eval_x, eval_y, batch_size=batch_size)

    predictions = model.predict(eval_x, batch_size=batch_size)
    auc = tf.metrics.auc(eval_y, predictions)

    print("Evaluation Loss:", loss[0])
    print("Accuracy:", loss[1])
    print("AUC: ", auc[0][1])


def predict_preamble(train_x_df, train_y_df):
    """Load the trained model and prepare a data point for prediction."""
    x = list(train_x_df.columns.values)
    model = build_model(len(x))

    try:
        model.load_weights(checkpoint_path)
    except errors.InvalidArgumentError:
        print("Weights couldn't be found, training before predicting")
        train(train_x_df, train_y_df)
        model = build_model(len(x))

    x = read_one_row(predict_row, train_x_df)

    return model, x


def predict(train_x_df, train_y_df):
    model, x = predict_preamble(train_x_df, train_y_df)

    print("Prediction:", model.predict(x)[0][0])


def benchmark(train_x_df, train_y_df):
    """Benchmark the time required to predict on the `bench_prediction` data."""
    model, x = predict_preamble(train_x_df, train_y_df)

    total_duration = 0
    for _ in range(0, bench_prediction):
        start = time.time()
        model.predict(x)
        end = time.time()
        duration = end - start

        total_duration = total_duration + duration

    print("Total Duration:", total_duration)
    print("Avg Runtime:", total_duration / bench_prediction * 1000, "ms")


def main():
    print("Home Credit Default!")

    # TODO only load all data when training
    train_x_df, train_y_df = data_prep(input_file)

    if predict_row != -1:
        predict(train_x_df, train_y_df)
    elif bench_prediction != -1:
        benchmark(train_x_df, train_y_df)
    else:
        train(train_x_df, train_y_df)


if __name__ == "__main__":
    main()
