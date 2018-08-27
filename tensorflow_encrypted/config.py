from typing import Dict, List, Optional, Any, Union, Tuple
from collections import defaultdict

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
from tensorflow.python.client import timeline
from abc import ABC, abstractmethod

from .player import Player


__TFE_DEBUG__ = False


class Config(ABC):
    @abstractmethod
    def players(self) -> List[Player]:
        pass

    @abstractmethod
    def get_player(self, name: str) -> Player:
        pass


class LocalConfig(Config):
    """
    Configure tf-encrypted to use threads on the local CPU
    to simulate the different players.
    Intended mostly for development/debugging use.
    """

    def __init__(self, player_names: List[str], job_name: str = 'localhost') -> None:
        player_names = ['master'] + player_names
        self._players = {
            name: Player(
                name=name,
                index=index,
                device_name='/job:{job_name}/task:0/device:CPU:{cpu_id}'.format(
                    job_name=job_name,
                    cpu_id=index
                )
            )
            for index, name in enumerate(player_names)
        }

    @property
    def players(self) -> List[Player]:
        return list(self._players.values())

    def get_player(self, name: str) -> Player:
        return self._players[name]

    def get_players(self, names: Union[List[str], str]) -> List[Player]:
        if isinstance(names, str):
            names = [name.strip() for name in names.split(',')]
        assert isinstance(names, list)
        return [player for name, player in self._players.items() if name in names]

    def session(self, log_device_placement: bool = False) -> tf.Session:
        global __TFE_DEBUG__
        # reserve one CPU for the player executing the script, to avoid
        # default pinning of operations to one of the actual players
        sess = tf.Session(
            '',
            None,
            config=tf.ConfigProto(
                log_device_placement=log_device_placement,
                allow_soft_placement=False,
                device_count={"CPU": len(self._players) + 1},
                inter_op_parallelism_threads=1,
                intra_op_parallelism_threads=1
            )
        )
        if __TFE_DEBUG__:
            print('session in debug mode')
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        return sess


class RemoteConfig(Config):
    """
    Configure tf-encrypted to use network hosts for the different players.
    """

    def __init__(self,
                 player_hostmap: Union[List[Tuple[str, str]], Dict[str, str]],
                 master_host: Optional[str] = '0.0.0.0:4440',
                 job_name: str = 'worker') -> None:

        self._job_name = job_name

        if isinstance(player_hostmap, dict):
            # ensure consistent ordering of the players across all instances
            player_hostmap = list(sorted(player_hostmap.items()))

        player_names, player_hosts = zip(*player_hostmap)
        player_names = ('master',) + player_names
        self._hostmap = (master_host,) + player_hosts
        self._target = 'grpc://{}'.format(master_host)

        self._players = {
            name: Player(
                name=name,
                index=index,
                device_name='/job:{job_name}/task:{task_id}/cpu:0'.format(
                    job_name=job_name,
                    task_id=index
                ),
                host=host
            )
            for index, (name, host) in enumerate(zip(player_names, self._hostmap))
        }

    @property
    def players(self) -> List[Player]:
        return list(self._players.values())

    def get_player(self, name: str) -> Player:
        return self._players[name]

    def get_players(self, names: Union[List[str], str]) -> List[Player]:
        if isinstance(names, str):
            names = [name.strip() for name in names.split(',')]
        assert isinstance(names, list)
        return [player for name, player in self._players.items() if name in names]

    def server(self, name: str) -> tf.train.Server:
        player = self.get_player(name)
        cluster = tf.train.ClusterSpec({self._job_name: self._hostmap})
        server = tf.train.Server(
            cluster,
            job_name=self._job_name,
            task_index=player.index
        )
        self._target = server.target
        print("Hi I'm node %s listening on %s with device_name %s" % (name, server.target, player.device_name))
        return server

    def session(self, log_device_placement: bool = False) -> tf.Session:
        global __TFE_DEBUG__

        config = tf.ConfigProto(
            log_device_placement=log_device_placement,
            allow_soft_placement=False,
        )
        print('Starting session on target: %s' % self._target, config)
        sess = tf.Session(
            self._target,
            config=config
        )
        if __TFE_DEBUG__:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        return sess


TENSORBOARD_DIR = '/tmp/tensorboard'
IGNORE_STATS = False
DEBUG = True

_run_counter: Any = defaultdict(int)


def setDebugMode(debug=True):
    global __TFE_DEBUG__
    if debug is True:
        print("Tensorflow encrypted is running in DEBUG mode")

    __TFE_DEBUG__ = debug


def run(
    sess: tf.Session,
    fetches: Any,
    feed_dict: Dict[str, np.ndarray] = {},
    tag: Optional[str] = None
) -> Any:

    if not DEBUG and (tag is None or IGNORE_STATS):

        return sess.run(
            fetches,
            feed_dict=feed_dict
        )

    else:

        session_tag = '{}{}'.format(tag, _run_counter[tag])
        run_tag = TENSORBOARD_DIR + ('/' + session_tag)
        _run_counter[tag] += 1

        writer = tf.summary.FileWriter(run_tag, sess.graph)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        results = sess.run(
            fetches,
            feed_dict=feed_dict,
            options=run_options,
            run_metadata=run_metadata
        )

        writer.add_run_metadata(run_metadata, session_tag)
        chrome_trace = timeline.Timeline(run_metadata.step_stats).generate_chrome_trace_format()
        with open('{}/{}.ctr'.format(TENSORBOARD_DIR, session_tag), 'w') as f:
            f.write(chrome_trace)

        return results
