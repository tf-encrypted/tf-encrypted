"""for simulation only"""
from collections import OrderedDict
from functools import wraps
import logging
import os
from pathlib import Path
import re
import sys

import tensorflow as tf
import tensorflow_datasets as tfds
import tf_encrypted as tfe

logger = logging.getLogger('tf_encrypted')
DEFAULT_PATH_PREFIX = os.path.join(str(Path.home()), ".tfe_data")


def tfds_random_splitter(dataset_name, num_splits, validation_split, **loader_kwargs):
  # Get split_name, if available
  split_name = loader_kwargs.pop("split", "train")

  def build_split_str(x, y=None):
    nonlocal split_name
    if split_name is None:
      split_name = 'train'
    if y is None:
      return "{split}[:{np}%]".format(split=split_name, np=int(x))
    return "{split}[{cp}%:{np}%]".format(split=split_name, cp=int(x), np=int(y))

  # Figure out subsplit quantities
  if validation_split is not None:
    validation_split *= 100  # convert decimal to percentile
    main_subsplit = 100 - validation_split
    current_percentile = validation_split
  else:
    main_subsplit = 100
    current_percentile = 0
  per_subsplit =  main_subsplit // num_splits


  splits = [build_split_str(current_percentile)]
  for i in range(num_splits):
    next_percentile = current_percentile + per_subsplit
    splits.append(build_split_str(current_percentile, next_percentile))
    current_percentile = next_percentile
  return tfds.load(dataset_name, split=splits, **loader_kwargs)


def dict_to_tensors(feature_dict):
  """expects key-value of features that are already tensors"""

  ordered_features = list(feature_dict.items())

  def _lift_to_tf_string(x):
    return tf.constant(x, dtype=tf.string)

  feature_names = []
  feature_values = []
  feature_dtypes = []
  for tensor_name, tensor in ordered_features:
    name = _lift_to_tf_string(tensor_name)
    dtype = _lift_to_tf_string(tensor.dtype.name)

    feature_names.append(name)
    feature_values.append(tensor)
    feature_dtypes.append(dtype)

  return feature_names + feature_values + feature_dtypes

def from_simulated_dataset(data_pipeline_func):
  """Wraps a data_pipeline_func meant for a simulated TF Dataset."""
  pattern = re.compile(".*-dtype")

  def reconstruct_example(instance):
    example = tf.train.Example()
    example.ParseFromString(instance.numpy())

    features = {}
    feature_dtypes = {}
    for name, val in example.items():
      match = re.match(name, pattern)
      if match is None:
        dtype = instance["{}-dtype".format(name)]
        parsed_tensor = tf.io.parse_tensor(tensor, tf.as_dtype(dtype))
        features[name] = parsed_tensor
    return features


  @wraps(data_pipeline_func)
  def wrapped(*args, **kwargs):
    dataset_container = args[0]
    ds = getattr(dataset_container, "dataset")
    ds = ds.map(reconstruct_example)
    ds = setattr(dataset_container, "dataset", ds)
    return data_pipeline_func(*args, **kwargs)

  return wrapped


def serialize_example(*features):
  third = len(features) // 3
  feature_names = features[:third]
  feature_values = features[third:2 * third]
  feature_dtypes = features[2 * third:]

  feature = {}

  nested_iterator = zip(feature_names, zip(feature_values, feature_dtypes))

  for tensor_name, (tensor_val, tensor_dtype) in nested_iterator:
    feature_name = tensor_name.numpy()
    feature_dtype_name = "{}-dtype".format(feature_name.decode("utf-8"))

    tensor_string = tf.io.serialize_tensor(tensor_val)
    tensor_bytes = tf.train.BytesList(
        value=[tensor_string.numpy()],
    )
    dtype_bytes = tf.train.BytesList(value=[tensor_dtype.numpy()])

    tensor = tf.train.Feature(bytes_list=tensor_bytes)
    dtype = tf.train.Feature(bytes_list=dtype_bytes)

    feature[feature_name] = tensor
    feature[feature_dtype_name] = dtype

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

  return example_proto.SerializeToString()


def tf_serialize_example(*features):
  tf_string = tf.py_function(serialize_example, features, tf.string)
  return tf.reshape(tf_string, ())


def federate_dataset(
    dataset_or_name,
    data_owner_names,
    model_owner_name=None,
    splitter_fn=tfds_random_splitter,
    validation_split=None,
    data_root = None,
    **load_kwargs):
  """Helper to split a dataset between data owners.

  For experimental purposes only.

  dataset_or_name: A tf.data.Dataset object, or a string representing a
      registered dataset from the TF Datasets catalogue.
  data_owner_names: Player name for data owners.
  model_owner_name: Player name for model owner. If None, no validation.
  splitter_fn: specifies how to split the dataset between the data owners.
      Default is tfds_random_splitter which chooses random splits for each.
  validation_split: Optional float on the interval (0,1), proportion of data to
      keep on ModelOwner for validation. Defaults to None (i.e. 0).
  data_root: Path-like designating where to write TFRecords on the data owners.
      If None, defaults to {prefix}, which assumes all devices are using the
      same filesystem format as this one (i.e. Windows vs. Unix).
  kwargs: If dataset_or_name is a registered dataset name from TFDS, these get
      passed to a call to tfds.load in the splitter_fn. If it's a
      tf.data.Dataset object, these get passed to the splitter_fn that handles
      it.
  """.format(prefix=DEFAULT_PATH_PREFIX)

  assert ((validation_split is None and model_owner_name is None)
          or (validation_split < 1 and model_owner_name is not None)), (
              "Invalid values for validation_split and model_owner_name."
          )

  print(("Splitting dataset for {} data owners. This is for simulation use "
         "only.").format(len(data_owner_names)))

  split_name = load_kwargs.get("split", None)

  if isinstance(dataset_or_name, str):
    # Assume it's the name of a dataset in tfds
    all_dataset = None
    dataset_name = dataset_or_name
  elif isinstance(dataset_or_name, tf.data.Dataset):
    all_dataset = dataset_or_name
    dataset_name = None
  else:
    raise

  num_splits = len(data_owner_names)
  if all_dataset is None:
    datasets = splitter_fn(dataset_name,
                           num_splits,
                           validation_split,
                           **load_kwargs)
  elif dataset_name is None:
    datasets = splitter_fn(all_dataset,
                           num_splits,
                           validation_split,
                           **load_kwargs)

  # Define default filepath for writing TFRecords
  # TODO: TFRecord sharding
  data_path_prefix = data_root or DEFAULT_PATH_PREFIX
  os.makedirs(data_path_prefix, exist_ok=True)

  ds_name = dataset_name or "tf-dataset"
  filepath_factory_fn = lambda owner_name: os.path.join(
      data_path_prefix,
      "{dsn}:{split}:{owner}.tfrecord".format(
          dsn=ds_name, split=split_name, owner=owner_name,
      ),
  )

  # Get each party's Player
  # We need their hostnames to write to each grpc server
  tfe_config = tfe.get_config()

  if model_owner_name is not None:
    players = [tfe_config.get_player(model_owner_name)]
  else:
    players = []
  for do in data_owner_names:
    players.append(tfe_config.get_player(do))

  players_to_tfrecords = OrderedDict()
  for player, ds in zip(players, datasets):
    local_path = filepath_factory_fn(player.name)
    writer = tf.data.experimental.TFRecordWriter(local_path)
    ds = ds.map(dict_to_tensors)
    ds = ds.map(tf_serialize_example)
    writer.write(ds)
    players_to_tfrecords[player.name] = [local_path]
  return players_to_tfrecords
