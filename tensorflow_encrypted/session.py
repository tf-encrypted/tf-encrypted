import os
from typing import Dict, List, Optional, Any, Union
from collections import Iterable, defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python import debug as tf_debug

from .config import Config, RemoteConfig, get_config

__TFE_STATS__ = bool(os.getenv('TFE_STATS', False))
__TFE_TRACE__ = bool(os.getenv('TFE_TRACE', False))
__TFE_DEBUG__ = bool(os.getenv('TFE_DEBUG', False))
__TENSORBOARD_DIR__ = str(os.getenv('TFE_STATS_DIR', '/tmp/tensorboard'))

_run_counter = defaultdict(int)  # type: Any


class Session(tf.Session):
    """
    Wrap a Tensorflow Session
    """

    def __init__(
        self,
        graph: Optional[tf.Graph]=None,
        config: Optional[Config]=None
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

    def sanitize_fetches(self, fetches: Any) -> Union[List[Any], tf.Tensor, tf.Operation]:
        if not isinstance(fetches, Iterable) or isinstance(fetches, tf.Tensor):
            if not isinstance(fetches, tf.Tensor) and not isinstance(fetches, tf.Operation):
                # decoder = fetches
                # to_decode = fetches.value_on_0.from_decomposed(fetches.value_on_0.backing)
                # return decoder.prot._decode(to_decode, decoder.is_scaled)
                return fetches.value_on_0.backing
            else:
                return fetches

        sanitized_fetches: List[Any] = []
        for idx, fetch in enumerate(fetches):
            sanitized_fetches.append(self.sanitize_fetches(fetch))

        return sanitized_fetches

    def decode_fetches(self, fetches: Any, fetches_out: Any) -> List[Union[tf.Tensor, tf.Operation]]:
        if not isinstance(fetches, Iterable) or isinstance(fetches, tf.Tensor):
            if not isinstance(fetches, tf.Tensor) and not isinstance(fetches, tf.Operation):
                decoder = fetches
                to_decode = decoder.value_on_0.from_decomposed(fetches_out)
                return decoder.prot._decode(to_decode, decoder.is_scaled)
            else:
                return fetches_out

        for idx, fetch in enumerate(fetches):
            fetches_out[idx] = self.decode_fetches(fetch, fetches_out[idx])

        return fetches_out

    def run(
        self,
        fetches: Any,
        feed_dict: Dict[str, np.ndarray] = {},
        tag: Optional[str] = None,
        write_trace: bool = False
    ) -> Any:

        sanitized_fetches = self.sanitize_fetches(fetches)

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
                chrome_trace = timeline.Timeline(run_metadata.step_stats).generate_chrome_trace_format()
                with open('{}/{}.ctr'.format(__TENSORBOARD_DIR__, session_tag), 'w') as f:
                    f.write(chrome_trace)

        # return fetches_out
        return self.decode_fetches(fetches, fetches_out)


def setMonitorStatsFlag(monitor_stats: bool = False) -> None:
    global __TFE_STATS__
    if monitor_stats is True:
        print("Tensorflow encrypted is monitoring statistics for each session.run() call using a tag")

    __TFE_STATS__ = monitor_stats


def setTFEDebugFlag(debug: bool = False) -> None:
    global __TFE_DEBUG__
    if debug is True:
        print("Tensorflow encrypted is running in DEBUG mode")

    __TFE_DEBUG__ = debug
