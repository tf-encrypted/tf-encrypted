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


def from_simulated_dataset(data_pipeline_func):
  """Wraps a data_pipeline_func meant for a simulated TF Dataset."""

  def reconstruct_example(example_proto):
    feature_desc = {
      "image": tf.io.FixedLenFeature([], tf.string, ''),
      "label": tf.io.FixedLenFeature([], tf.string, ''),
    }
    example = tf.io.parse_single_example(example_proto, feature_desc)
    example["image"] = tf.io.parse_tensor(example["image"], tf.uint8)
    example["label"] = tf.io.parse_tensor(example["label"], tf.int64)
    return example

  @wraps(data_pipeline_func)
  def wrapped(*args, **kwargs):
    dataset_container = args[0]
    ds = getattr(dataset_container, "dataset")
    ds = ds.map(reconstruct_example)
    ds = setattr(dataset_container, "dataset", ds)
    return data_pipeline_func(*args, **kwargs)

  return wrapped


def serialize_image_example(instance):

  def _serializer(image, label):
    eager_type = type(tf.constant(0))

    im = tf.io.serialize_tensor(image)
    lbl = tf.io.serialize_tensor(label)
    if isinstance(im, eager_type):
      im = im.numpy()
      lbl = lbl.numpy()
    im_feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=[im]))
    lbl_feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=[lbl]))

    features = dict(image=im_feat, label=lbl_feat)
    ex_proto = tf.train.Example(features=tf.train.Features(feature=features))

    return ex_proto.SerializeToString()

  inputs = [instance['image'], instance['label']]
  tf_str = tf.py_function(_serializer, inputs, tf.string)
  return tf.reshape(tf_str, ())


def federate_dataset(
    dataset_or_name,
    data_owner_names,
    model_owner_name=None,
    splitter_fn=tfds_random_splitter,
    serializer_fn=serialize_image_example,
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
    ds = ds.map(serializer_fn)
    writer.write(ds)
    players_to_tfrecords[player.name] = [local_path]
  return players_to_tfrecords
