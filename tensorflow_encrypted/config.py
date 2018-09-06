import os
from typing import Dict, List, Optional, Any, Union, Tuple
from collections import defaultdict

import json
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
from tensorflow.python.client import timeline
from abc import ABC, abstractmethod

from .player import Player

__TFE_DEBUG__ = False
TENSORBOARD_DIR = '/tmp/tensorboard'
__MONITOR_STATS__ = False

_run_counter: Any = defaultdict(int)


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

    def __init__(self, player_names: List[str], job_name: str='localhost') -> None:
        self._job_name = job_name
        self._players = {
            name: Player(
                name=name,
                index=index,
                device_name='/job:{job_name}/replica:0/task:0/device:CPU:{cpu_id}'.format(
                    job_name=job_name,
                    cpu_id=index
                )
            )
            for index, name in enumerate(player_names)
        }

    @staticmethod
    def from_dict(params) -> Optional['LocalConfig']:
        if not params.get('type', None) == 'local':
            return None

        return LocalConfig(
            player_names=params['player_names'],
            job_name=params['job_name']
        )

    def to_dict(self) -> Dict:
        params = {
            'type': 'local',
            'job_name': self._job_name,
            'player_names': [p.name for p in sorted(self._players.values(), key=lambda p: p.index)]
        }
        return params

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

    def session(
        self,
        master: Optional[Union[int, str]] = None,
        log_device_placement: bool = False
    ) -> tf.Session:    

        if master is not None:
            print("WARNING: master '{}' is ignored, always using first player".format(master))

        sess = tf.Session(
            '',
            None,
            config=tf.ConfigProto(
                log_device_placement=log_device_placement,
                allow_soft_placement=False,
                device_count={"CPU": len(self._players)},
                inter_op_parallelism_threads=1,
                intra_op_parallelism_threads=1
            )
        )

        global __TFE_DEBUG__
        if __TFE_DEBUG__:
            print('Session in debug mode')
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        return sess


class RemoteConfig(Config):
    """
    Configure tf-encrypted to use network hosts for the different players.
    """

    def __init__(self,
                 hostmap: Union[List[Tuple[str, str]], Dict[str, str]],
                 master: Optional[Union[int, str]] = None,
                 job_name: str = 'tfe') -> None:

        if isinstance(hostmap, dict):
            # ensure consistent ordering of the players across all instances
            hostmap = list(sorted(hostmap.items()))

        self._job_name = job_name
        self._master = master
        self._hostmap = hostmap
        self._players = {
            name: Player(
                name=name,
                index=index,
                device_name='/job:{job_name}/replica:0/task:{task_id}/cpu:0'.format(
                    job_name=job_name,
                    task_id=index
                ),
                host=host
            )
            for index, (name, host) in enumerate(self._hostmap)
        }

    @staticmethod
    def from_dict(params) -> Optional['RemoteConfig']:
        if not params.get('type', None) == 'remote':
            return None

        return RemoteConfig(
            hostmap=params['hostmap'],
            job_name=params['job_name'],
            master=params.get('master', None)
        )

    def to_dict(self) -> Dict:
        params = {
            'type': 'remote',
            'job_name': self._job_name,
            'hostmap': [(p.name, p.host) for p in sorted(self._players.values(), key=lambda p: p.index)]
        }

        if self._master is not None:
            params['master'] = self._master

        return params

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
        server = tf.train.Server(cluster, job_name=self._job_name, task_index=player.index)
        print("Hi, I'm node '{name}' running as device '{device}' and with session target '{target}'".format(
            name=name,
            device=player.device_name,
            target=server.target
        ))
        return server

    def _compute_target(self, master: Optional[Union[int, str]]):

        # if no specific master is given then fall back to the default
        if master is None:
            # ... and if no default master is given then use first player
            master = self._master if self._master is not None else 0

        if isinstance(master, int):
            # interpret as index
            master_host = self._hostmap[master][1]

        elif isinstance(master, str):
            # is it a player name?
            player = self._players.get(master, None)
            if player is not None:
                # ... yes it was so use its host
                master_host = player.host
            else:
                # ... no it wasn't so just assume it's an URI
                master_host = master

        else:
            raise TypeError("Don't know how to turn '{}' into a target".format(master))

        target = 'grpc://{}'.format(master_host)
        return target

    def session(
        self,
        master: Optional[Union[int, str]] = None,
        log_device_placement: bool = False
    ) -> tf.Session:

        target = self._compute_target(master)
        config = tf.ConfigProto(
            log_device_placement=log_device_placement,
            allow_soft_placement=False,
        )
        print("Starting session on target '{}' using config {}".format(target, config))
        sess = tf.Session(target, config=config)

        global __TFE_DEBUG__
        if __TFE_DEBUG__:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        return sess


def load(filename) -> Optional[Config]:
    with open(filename, 'r') as f:
        params = json.load(f)

    for config_cls in [LocalConfig, RemoteConfig]:
        config = config_cls.from_dict(params)
        if config is not None:
            return config

    return None


def save(config, filename) -> None:
    with open(filename, 'w') as f:
        json.dump(config.to_dict(), f)


TENSORBOARD_DIR = '/tmp/tensorboard'
__MONITOR_STATS__ = False

_run_counter: Any = defaultdict(int)


def setTFEDebugFlag(debug: bool = False) -> None:
    global __TFE_DEBUG__
    if debug is True:
        print("Tensorflow encrypted is running in DEBUG mode")

    __TFE_DEBUG__ = debug


def setMonitorStatsFlag(monitor_stats: bool = False) -> None:
    global __MONITOR_STATS__
    if monitor_stats is True:
        print("Tensorflow encrypted is monitoring statistics for each session.run() call using a tag")

    __MONITOR_STATS__ = monitor_stats


def run(
    sess: tf.Session,
    fetches: Any,
    feed_dict: Dict[str, np.ndarray] = {},
    tag: Optional[str] = None
) -> Any:
    if __MONITOR_STATS__ and tag is not None:
        session_tag = "{}{}".format(tag, _run_counter[tag])
        run_tag = os.path.join(TENSORBOARD_DIR, session_tag)
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

        writer.add_run_metadata(run_metadata, 'step0')
        writer.close()

        chrome_trace = timeline.Timeline(run_metadata.step_stats).generate_chrome_trace_format()
        with open('{}/{}.ctr'.format(TENSORBOARD_DIR, session_tag), 'w') as f:
            f.write(chrome_trace)

        return results
    else:
        return sess.run(
            fetches,
            feed_dict=feed_dict
        )
