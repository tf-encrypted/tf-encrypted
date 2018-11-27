import os
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python import debug as tf_debug

from .config import RemoteConfig, get_config
from .protocol.pond import PondPublicTensor
from .tensor.factory import AbstractTensor


__TFE_STATS__ = bool(os.getenv('TFE_STATS', False))
__TFE_TRACE__ = bool(os.getenv('TFE_TRACE', False))
__TFE_DEBUG__ = bool(os.getenv('TFE_DEBUG', False))
__TENSORBOARD_DIR__ = str(os.getenv('TFE_STATS_DIR', '/tmp/tensorboard'))

_run_counter = defaultdict(int)  # type: Any


class Session(tf.Session):
    """
    Wrap a Tensorflow Session.

    See the documentation of
    `tf.Session <https://www.tensorflow.org/api_docs/python/tf/Session>`_
    for more details.

    :param Optional[tf.Graph] graph: A :class:`tf.Graph`.  Used in the same as in tensorflow.
            This is the graph to be launched.  If nothing is specified then the default session
            graph will be used.
    :param Optional[~tensorflow_encrypted.config.Config] config:  A
            :class:`Local <tf_encrypted.config.LocalConfig/>` or
            :class:`Remote <tensorflow_encrypted.config.RemoteConfig>` config to be used to
            execute the graph.
    """

    def __init__(
        self,
        graph=None,
        config=None
    ) -> None:
        if config is None:
            config = get_config()

        target, configProto = config.get_tf_config()

        if isinstance(config, RemoteConfig):
            print("Starting session on target '{}' using config {}".format(target, configProto))
        super(Session, self).__init__(target, graph, configProto)
        # self.sess = tf.Session(target, graph, configProto)

        global __TFE_DEBUG__
        if __TFE_DEBUG__:
            print('Session in debug mode')
            self = tf_debug.LocalCLIDebugWrapperSession(self)

    def _sanitize_fetches(self, fetches: Any) -> Union[List[Any], tf.Tensor, tf.Operation]:

        if isinstance(fetches, (list, tuple)):
            return [self._sanitize_fetches(fetch) for fetch in fetches]
        else:
            if isinstance(fetches, (tf.Tensor, tf.Operation)):
                return fetches
            elif isinstance(fetches, PondPublicTensor):
                return fetches.decode()
            elif isinstance(fetches, AbstractTensor):
                return fetches.to_native()
            else:
                raise TypeError("Don't know how to fetch {}", type(fetches))

    def run(
        self,
        fetches: Any,
        feed_dict: Dict[tf.Tensor, np.ndarray] = {},
        tag: Optional[str] = None,
        write_trace: bool = False
    ):
        """
        run(fetches, feed_dict, tag, write_trace) -> Any

        See the documentation for
        `tf.Session.run <https://www.tensorflow.org/api_docs/python/tf/Session#run>`_
        for more details.

        This method functions just as the one from tensorflow.

        The value returned by run() has the same shape as the fetches argument, where the leaves
        are replaced by the corresponding values returned by TensorFlow.

        :param Any fetches: A single graph element, a list of graph elements, or a dictionary whose
                            values are graph elements or lists of graph elements
                            (described in tf.Session.run docs).
        :param str->np.ndarray feed_dict: A dictionary that maps graph elements to values
                                          (described in tf.Session.run docs).
        :param str tag: An optional namespace to run the session under.
        :param bool write_trace: If true, the session logs will be dumped for use in Tensorboard.
        """

        sanitized_fetches = self._sanitize_fetches(fetches)

        if not __TFE_STATS__ or tag is None:
            fetches_out = super(Session, self).run(
                sanitized_fetches,
                feed_dict=feed_dict
            )
        else:
            session_tag = "{}{}".format(tag, _run_counter[tag])
            run_tag = os.path.join(__TENSORBOARD_DIR__, session_tag)
            _run_counter[tag] += 1

            writer = tf.summary.FileWriter(run_tag, self.graph)
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            fetches_out = super(Session, self).run(
                sanitized_fetches,
                feed_dict=feed_dict,
                options=run_options,
                run_metadata=run_metadata
            )

            writer.add_run_metadata(run_metadata, session_tag)
            writer.close()

            if __TFE_TRACE__ or write_trace:
                tracer = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = tracer.generate_chrome_trace_format()
                with open('{}/{}.ctr'.format(__TENSORBOARD_DIR__, session_tag), 'w') as f:
                    f.write(chrome_trace)

        return fetches_out


def setMonitorStatsFlag(monitor_stats: bool = False) -> None:
    """
    setMonitorStatsFlag(monitor_stats)

    Set flag to enable or disable monitoring of runtime statistics for each call to session.run().

    :param bool monitor_stats: Enable or disable stats, disabled by default.
    """
    global __TFE_STATS__
    if monitor_stats is True:
        print("Tensorflow encrypted is monitoring statistics for each session.run() \
              call using a tag")

    __TFE_STATS__ = monitor_stats


def setTFEDebugFlag(debug: bool = False) -> None:
    """
    setTFEDebugFlag(debug)

    Set flag to enable or disable debugging mode for tf-encrypted.

    :param bool debug: Enable or disable debugging, disabled by default.
    """
    global __TFE_DEBUG__
    if debug is True:
        print("Tensorflow encrypted is running in DEBUG mode")

    __TFE_DEBUG__ = debug


def setTFETraceFlag(trace: bool = False) -> None:
    """
    setTFETraceFlag(trace)

    Set flag to enable or disable tracing in tf-encrypted.

    :param bool trace: Enable or disable tracing, disabled by default.
    """
    global __TFE_TRACE__
    if trace is True:
        print("Tensorflow encrypted is dumping computation traces")

    __TFE_TRACE__ = trace
