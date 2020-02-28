"""CLI for data preparation and processing."""
import argparse

from utils import data_prep
from utils import read_one_row
from utils import save_input

parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_row",
    type=int,
    default="0",
    help="Saves a single row to a file defaults to row 0",
)
parser.add_argument(
    "--input_file",
    type=str,
    default="final_data_with_feature_engineered.csv",
    help=(
        "File to read the row from defaults to "
        "final_data_with_feature_engineered.csv"
    ),
)
parser.add_argument(
    "--output_file",
    type=str,
    default="input.npy",
    help=("Output file with the input row defaults to " "input.npy"),
)
config = parser.parse_args()

input_file = config.input_file
output_file = config.output_file
save_row = config.save_row

train_x_df, _ = data_prep(input_file)
out = read_one_row(save_row, train_x_df)
save_input(output_file, out)
