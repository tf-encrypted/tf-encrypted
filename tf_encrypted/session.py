"""TF Encrypted extension of tf.Session."""
import os
from collections import defaultdict
import logging

import tensorflow as tf
from tensorflow.python.client import timeline

from .config import RemoteConfig, get_config
from .utils import unwrap_fetches


# pylint: disable=invalid-name
__TFE_EVENTS__ = bool(os.getenv('TFE_EVENTS', ""))
__TFE_TRACE__ = bool(os.getenv('TFE_TRACE', ""))
__TENSORBOARD_DIR__ = str(os.getenv('TFE_EVENTS_DIR', '/tmp/tensorboard'))
# pylint: enable=invalid-name

_run_counter = defaultdict(int)

logging.basicConfig()
logger = logging.getLogger('tf_encrypted')
logger.setLevel(logging.DEBUG)


class Session(tf.Session):
  """
  Wrap a Tensorflow Session.

  See the documentation of
  `tf.Session <https://www.tensorflow.org/api_docs/python/tf/Session>`_
  for more details.

  :param Optional[tf.Graph] graph: A :class:`tf.Graph`.  Used similarly.
    This is the graph to be launched.  If nothing is specified, then the
    default graph in the session will be used.
  :param Optional[~tensorflow_encrypted.config.Config] config:  A
    :class:`Local <tf_encrypted.config.LocalConfig/>` or
    :class:`Remote <tf_encrypted.config.RemoteConfig>` config to be supplied
    when executing the graph.
  """

  def __init__(self, graph=None, config=None, target=None, **tf_config_kwargs):
    if config is None:
      config = get_config()

    default_target, config_proto = config.get_tf_config(**tf_config_kwargs)
    if target is None:
      target = default_target
    self.target = target
    self.config_proto = config_proto

    if isinstance(config, RemoteConfig):
      logger.info("Starting session on target '%s' using config %s",
                  self.target, self.config_proto)
    super(Session, self).__init__(self.target, graph, self.config_proto)

  def run(
      self,
      fetches,
      feed_dict=None,
      tag=None,
      write_trace=False,
      output_partition_graphs=False
  ):
    # pylint: disable=arguments-differ
    """
    run(fetches, feed_dict, tag, write_trace) -> Any

    See the documentation for
    `tf.Session.run <https://www.tensorflow.org/api_docs/python/tf/Session#run>`_
    for more details.

    This method functions just as the one from tensorflow.

    The value returned by run() has the same shape as the fetches argument,
    where the leaves are replaced by the corresponding values returned by
    TensorFlow.

    :param Any fetches: A single graph element, a list of graph elements, or a
      dictionary whose values are graph elements or lists of graph elements
      (described in tf.Session.run docs).
    :param str->np.ndarray feed_dict: A dictionary that maps graph elements to
      values (described in tf.Session.run docs).
    :param str tag: An optional namespace to run the session under.
    :param bool write_trace: If true, the session logs will be dumped for use
      in Tensorboard.
    """

    sanitized_fetches = unwrap_fetches(fetches)

    if not __TFE_EVENTS__ or tag is None:
      fetches_out = super(Session, self).run(
          sanitized_fetches,
          feed_dict=feed_dict
      )
    else:
      session_tag = "{}{}".format(tag, _run_counter[tag])
      run_tag = os.path.join(__TENSORBOARD_DIR__, session_tag)
      _run_counter[tag] += 1

      writer = tf.summary.FileWriter(run_tag, self.graph)
      run_options = tf.RunOptions(
          trace_level=tf.RunOptions.FULL_TRACE,
          output_partition_graphs=output_partition_graphs
      )

      run_metadata = tf.RunMetadata()

      fetches_out = super(Session, self).run(
          sanitized_fetches,
          feed_dict=feed_dict,
          options=run_options,
          run_metadata=run_metadata
      )

      if output_partition_graphs:
        for i, g in enumerate(run_metadata.partition_graphs):
          tf.io.write_graph(
              g,
              logdir=os.path.join(__TENSORBOARD_DIR__, session_tag),
              name='partition{}.pbtxt'.format(i),
          )

      writer.add_run_metadata(run_metadata, session_tag)
      writer.close()

      if __TFE_TRACE__ or write_trace:
        tracer = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = tracer.generate_chrome_trace_format()
        trace_fname = '{}/{}.ctr'.format(__TENSORBOARD_DIR__, session_tag)
        with open(trace_fname, 'w') as f:
          f.write(chrome_trace)

    return fetches_out


def set_tfe_events_flag(monitor_events: bool = False) -> None:
  """
  set_tfe_events_flag(monitor_events)

  Set flag to enable or disable monitoring of runtime statistics for each call
  to session.run().

  :param bool monitor_events: Enable or disable stats, disabled by default.
  """
  global __TFE_EVENTS__  # pylint: disable=invalid-name
  if monitor_events is True:
    logger.info("Writing event files for each run with a tag")

  __TFE_EVENTS__ = monitor_events


def set_tfe_trace_flag(trace: bool = False) -> None:
  """
  set_tfe_trace_flag(trace)

  Set flag to enable or disable tracing in TF Encrypted.

  :param bool trace: Enable or disable tracing, disabled by default.
  """
  global __TFE_TRACE__  # pylint: disable=invalid-name
  if trace is True:
    logger.info("Writing trace files")

  __TFE_TRACE__ = trace


def set_log_directory(path):
  """
  set_log_directory(path)

  Sets the directory to write TensorBoard event and trace files to.

  :param str path: The TensorBoard logdir.
  """
  global __TENSORBOARD_DIR__  # pylint: disable=invalid-name
  if path:
    logger.info("Writing event and trace files to '%s'", path)

  __TENSORBOARD_DIR__ = path
