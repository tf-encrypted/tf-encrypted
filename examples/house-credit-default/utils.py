"""Data preparation utilities."""
import numpy as np

import pandas as pd


def data_prep(filename):
    """Preprocess data from CSV."""
    df = pd.read_csv(filename)

    # remove whitespace from pandas names
    df.columns = df.columns.str.replace(" ", "_")
    df.columns = df.columns.str.replace(":", "-")
    df.columns = df.columns.str.replace("(", "_")
    df.columns = df.columns.str.replace(")", "_")
    df.columns = df.columns.str.replace("+", ".")
    df.columns = df.columns.str.replace(",", "_")

    train_df = df[df["TARGET"].notnull()]
    train_x_df = train_df.drop(columns=["TARGET", "SK_ID_CURR", "index"])
    train_y_df = train_df["TARGET"]

    return train_x_df, train_y_df


def save_input(filename, output):
    np.save(filename, output)


def read_one_row(row, train_x_df):
    return train_x_df[row : row + 1]
