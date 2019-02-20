from abc import ABC, abstractmethod
from collections import OrderedDict
import json
import logging
import math
from pathlib import Path

import tensorflow as tf

from .player import Player


logging.basicConfig()
logger = logging.getLogger('tf_encrypted')
logger.setLevel(logging.DEBUG)


def tensorflow_supports_int64():
    # hacky way to test if int64 is supported by this build of TensorFlow
    with tf.Graph().as_default():
        x = tf.constant([1], shape=(1, 1), dtype=tf.int64)
        try:
            tf.matmul(x, x)
        except TypeError:
            return False
        return True


def _get_docker_cpu_quota():
    # If you witness memory leaks while doing multiple predictions using docker
    # see https://github.com/tensorflow/tensorflow/issues/22098

    cpu_cores = None

    # Check for quotas if we are in a linux container
    cfs_period = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    cfs_quota = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")

    if cfs_period.exists() and cfs_quota.exists():
        with cfs_period.open('rb') as p, cfs_quota.open('rb') as q:
            p_int, q_int = int(p.read()), int(q.read())

            # get the cores allocated by dividing the quota
            # in microseconds by the period in microseconds
            if q_int > 0 and p_int > 0:
                cpu_cores = math.ceil(q_int / p_int)

    return cpu_cores


class Config(ABC):

    @abstractmethod
    def players(self):
        """
        Returns the config's list of :class:`Player` objects.
        """
        pass

    @abstractmethod
    def get_player(self, name):
        """
        Retrieve a specific :class:`Player` object by name.
        """
        pass

    @abstractmethod
    def get_tf_config(self):
        """
        get_tf_config() -> tf.ConfigProto, or str

        Extract the underlying :class:`tf.ConfigProto`.
        """
        pass


class LocalConfig(Config):
    """
    Configure TF Encrypted to use threads on the local CPU to simulate the different players.

    Intended mostly for development/debugging use.

    By default new players will be added when looked up for the first time;
    this is useful for  instance to get a complete list of players involved
    in a particular computation (see `auto_add_unknown_players`).

    :param (str) player_names: List of players to be used in the session.
    :param str job_name: The name of the job.
    :param bool auto_add_unknown_players: Automatically add player on first lookup.
    """

    def __init__(
        self,
        player_names=[],
        job_name='localhost',
        auto_add_unknown_players=True,
    ) -> None:
        self._job_name = job_name
        self._auto_add_unknown_players = auto_add_unknown_players
        self._players = []
        for name in player_names:
            self.add_player(name)

    def add_player(self, name):
        index = len(self._players)
        player = Player(
            name=name,
            index=index,
            device_name='/job:{job_name}/replica:0/task:0/device:CPU:{cpu_id}'.format(
                job_name=self._job_name,
                cpu_id=index,
            )
        )
        self._players.append(player)
        return player

    @property
    def players(self):
        return self._players

    def get_player(self, name):
        player = next((player for player in self._players if player.name == name), None)
        if player is None and self._auto_add_unknown_players:
            player = self.add_player(name)
        return player

    def get_players(self, names):
        if isinstance(names, str):
            names = [name.strip() for name in names.split(',')]
        assert isinstance(names, list)
        return [player for player in self._players if player.name in names]

    def get_tf_config(self, log_device_placement=False):
        logger.info("Players: {}".format([player.name for player in self.players]))
        target = ''
        config = tf.ConfigProto(
            log_device_placement=log_device_placement,
            allow_soft_placement=False,
            device_count={"CPU": len(self._players)}
        )
        return (target, config)


class RemoteConfig(Config):
    """
    Configure TF Encrypted to use network hosts for the different players.

    :param (str,str),str->str hostmap: A mapping of hostnames to
        their IP / domain.
    :param str job_name: The name of the job.
    """

    def __init__(
        self,
        hostmap,
        job_name='tfe',
    ):
        assert isinstance(hostmap, dict)
        if not isinstance(hostmap, OrderedDict):
            logger.warning(
                "Consider passing an ordered dictionary to RemoteConfig instead"
                "in order to preserve host mapping.")

        self._job_name = job_name
        self._players = OrderedDict(
            (name, Player(
                name=name,
                index=index,
                device_name='/job:{job_name}/replica:0/task:{task_id}/cpu:0'.format(
                    job_name=job_name,
                    task_id=index
                ),
                host=host
            ))
            for index, (name, host) in enumerate(hostmap.items())
        )

    @staticmethod
    def load(filename):
        """
        Constructs a RemoteConfig object from a JSON hostmap file.

        :param str filename: Name of file to load from.
        """
        with open(filename, 'r') as f:
            hostmap = json.load(f, object_pairs_hook=OrderedDict)
        return RemoteConfig(hostmap)

    def save(self, filename):
        """
        Saves the configuration as a JSON hostmap file.

        :param str filename: Name of file to save to.
        """
        with open(filename, 'w') as f:
            json.dump(self.hostmap, f)

    @property
    def hostmap(self):
        return OrderedDict(
            (player.name, player.host)
            for player in self._players.values()
        )

    @property
    def hosts(self):
        return [
            player.host
            for player in self._players.values()
        ]

    @property
    def players(self):
        return list(self._players.values())

    def get_player(self, name):
        return self._players.get(name)

    def get_players(self, names):
        if isinstance(names, str):
            names = [name.strip() for name in names.split(',')]
        assert isinstance(names, list)
        return [player for name, player in self._players.items() if name in names]

    def server(self, name, start=True):
        """
        Construct a :class:`tf.train.Server` object for the corresponding :class:`Player`.

        :param str name: Name of player.
        """
        player = self.get_player(name)
        assert player is not None, "'{}' not found in configuration".format(name)
        cluster = tf.train.ClusterSpec({self._job_name: self.hosts})
        logger.debug("Creating server for '{name}' using {cluster}".format(
            name=name,
            cluster=cluster))
        server = tf.train.Server(
            cluster,
            job_name=self._job_name,
            task_index=player.index,
            start=start)
        logger.info("Created server for '{name}' as device '{device}'; "
                    "own session target is '{target}'".format(
                        name=name,
                        device=player.device_name,
                        target=server.target))
        return server

    def get_tf_config(self, log_device_placement=False):
        # always use the first host as master; change config to match
        target = 'grpc://{}'.format(self.hosts[0])
        cpu_cores = _get_docker_cpu_quota()
        if cpu_cores is None:
            config = tf.ConfigProto(
                log_device_placement=log_device_placement,
                allow_soft_placement=False
            )
        else:
            config = tf.ConfigProto(
                log_device_placement=log_device_placement,
                allow_soft_placement=False,
                inter_op_parallelism_threads=cpu_cores,
                intra_op_parallelism_threads=cpu_cores
            )
        return (target, config)


__CONFIG__ = LocalConfig()


def get_config():
    """
    Returns the current config.
    """
    return __CONFIG__


def set_config(config) -> None:
    """
    Sets the current config.

    :param Config config: Intended configuration.
    """
    global __CONFIG__
    __CONFIG__ = config
