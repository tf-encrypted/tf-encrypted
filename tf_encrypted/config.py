import json
import math
from typing import Dict, List, Optional, Union, Tuple
from abc import ABC, abstractmethod
from pathlib import Path

import tensorflow as tf

from .player import Player


def tensorflow_supports_int64() -> bool:
    # hacky way to test if int64 is supported by this build of TensorFlow
    with tf.Graph().as_default():
        x = tf.constant([1], shape=(1, 1), dtype=tf.int64)
        try:
            tf.matmul(x, x)
        except TypeError:
            return False
        return True


def _get_docker_cpu_quota() -> Optional[int]:
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
    def players(self) -> List[Player]:
        """
        players() -> List

        Returns the config's list of :class:`Player` objects.
        """
        pass

    @abstractmethod
    def get_player(self, name: str) -> Player:
        """
        get_player(name) -> Player

        Retrieve a specific :class:`Player` object by name.
        """
        pass

    @abstractmethod
    def from_dict(cls, params):
        pass

    @abstractmethod
    def to_dict(self) -> Dict:
        """
        to_dict() -> Dict

        Writes the config to a dictionary.
        """

    @abstractmethod
    def get_tf_config(self) -> Tuple[str, tf.ConfigProto]:
        """
        get_tf_config() -> tf.ConfigProto, or str

        Extract the underlying :class:`tf.ConfigProto`.
        """
        pass


class LocalConfig(Config):
    """
    LocalConfig(player_names, master=None, job_name='localhost', log_device_placement=False)

    Configure tf-encrypted to use threads on the local CPU
    to simulate the different players.
    Intended mostly for development/debugging use.

    :param (str) player_names: List of players to be used in the session.
    :param int,str master: Optional pointer to the master node.
        If `int`, denotes the index of the master's name in the `player_names`.
        If `str`, denotes the player name.
    :param str job_name: The name of the job.
    :param bool log_device_placement: Whether or not to write device placement in logs.
    """

    def __init__(
        self,
        player_names,
        master=None,
        job_name='localhost',
        log_device_placement=False
    ) -> None:
        self._master = master
        self._log_device_placement = log_device_placement
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

    @classmethod
    def from_dict(cls, params: Dict) -> Optional['LocalConfig']:
        """
        from_dict(params) -> LocalConfig

        Produces a LocalConfig class from a dictionary.

        :param dict params: Key-value store of constructor arguments.
        """
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

    def get_tf_config(self) -> Tuple[str, tf.ConfigProto]:
        if self._master is not None:
            print("WARNING: master '{}' is ignored, always using first player".format(self._master))

        target = ''
        config = tf.ConfigProto(
            log_device_placement=self._log_device_placement,
            allow_soft_placement=False,
            device_count={"CPU": len(self._players)}
        )

        return (target, config)


class RemoteConfig(Config):
    """
    RemoteConfig(hostmap, master=None, job_name='tfe', log_device_placement=False)

    Configure tf-encrypted to use network hosts for the different players.

    :param (str,str),str->str hostmap: A mapping of hostnames to
        their IP / domain.
    :param int,str master: Optional pointer to the master node.
        If `int`, denotes the index of the master name in the hostmap (alphabetical if dict).
        If `str`, denotes the player name or node URI.
    :param str job_name: The name of the job.
    :param bool log_device_placement: Whether or not to write device placement in logs.
    """

    def __init__(
        self,
        hostmap,
        master=None,
        job_name='tfe',
        log_device_placement=False
    ) -> None:

        if isinstance(hostmap, dict):
            # ensure consistent ordering of the players across all instances
            hostmap = list(sorted(hostmap.items()))

        self._job_name = job_name
        self._master = master
        self._log_device_placement = log_device_placement
        self._hosts = [entry[1] for entry in hostmap]
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
            for index, (name, host) in enumerate(hostmap)
        }

    @classmethod
    def from_dict(cls, params: Dict) -> Optional['RemoteConfig']:
        """
        from_dict(params) -> RemoteConfig

        Produces a RemoteConfig class from a dictionary.

        :param dict params: Key-value store of constructor arguments.
        """
        if not params.get('type', None) == 'remote':
            return None

        return RemoteConfig(
            hostmap=params['hostmap'],
            job_name=params['job_name'],
            master=params.get('master', None)
        )

    def to_dict(self) -> Dict:
        hmap = [(p.name, p.host) for p in sorted(self._players.values(), key=lambda p: p.index)]
        params = {
            'type': 'remote',
            'job_name': self._job_name,
            'hostmap': hmap
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
        """
        server(name) -> tf.train.Server

        Retrieves the :class:`tf.train.Server` object from the :class:`RemoteConfig` by the
        corresponding :class:`Player` name.

        :param str name: Name of the server's corresponding player.
        """
        player = self.get_player(name)
        cluster = tf.train.ClusterSpec({self._job_name: self._hosts})
        server = tf.train.Server(cluster, job_name=self._job_name, task_index=player.index)
        print("Hi, I'm node '{name}' running as device '{device}' \
               and with session target '{target}'".format(
            name=name,
            device=player.device_name,
            target=server.target
        ))
        return server

    def _compute_target(self, master: Optional[Union[int, str]]) -> str:

        # if no specific master is given then fall back to the default
        if master is None:
            # ... and if no default master is given then use first player
            master = self._master if self._master is not None else 0

        if isinstance(master, int):
            # interpret as index
            master_host = self._hosts[master]

        elif isinstance(master, str):
            # is it a player name?
            player = self._players.get(master, None)
            if player is not None and player.host is not None:
                # ... yes it was so use its host
                master_host = player.host
            else:
                # ... no it wasn't so just assume it's an URI
                master_host = master

        else:
            raise TypeError("Don't know how to turn '{}' into a target".format(master))

        target = 'grpc://{}'.format(master_host)
        return target

    def get_tf_config(self) -> Tuple[str, tf.ConfigProto]:
        cpu_cores = _get_docker_cpu_quota()
        target = self._compute_target(self._master)
        # If you witness memory leaks while doing multiple predictions using docker
        # see https://github.com/tensorflow/tensorflow/issues/22098
        if cpu_cores is None:
            config = tf.ConfigProto(
                log_device_placement=self._log_device_placement,
                allow_soft_placement=False
            )
        else:
            config = tf.ConfigProto(
                log_device_placement=self._log_device_placement,
                allow_soft_placement=False,
                inter_op_parallelism_threads=cpu_cores,
                intra_op_parallelism_threads=cpu_cores
            )

        return (target, config)


def load(filename: str) -> Config:
    """
    load(filename) -> Config

    Constructs a Config object from a json file.
    """
    with open(filename, 'r') as f:
        params = json.load(f)

    config_type = params.get('type', None)
    if config_type == 'remote':
        return RemoteConfig.from_dict(params)
    elif config_type == 'local':
        return LocalConfig.from_dict(params)

    raise ValueError("Failed to parse config file")


def save(config: Config, filename: str) -> None:
    """
    save(config, filename)

    Saves a Config object as a json file.

    :param Config config: The Config object to save.
    :param str filename: Name of the intended json file.
    """
    with open(filename, 'w') as f:
        json.dump(config.to_dict(), f)


__CONFIG__ = LocalConfig([
    'input-provider',
    'model-provider',
    'server0',
    'server1',
    'crypto-producer'
])


def get_config():
    """
    get_config() -> Config

    Returns the current config.
    """
    return __CONFIG__


def set_config(config: Config) -> None:
    """
    set_config(config)

    Sets the current config.

    :param Config config: Intended Config object.
    """
    global __CONFIG__
    __CONFIG__ = config
