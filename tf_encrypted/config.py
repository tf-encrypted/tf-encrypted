"""The TF Encrypted Config abstraction and its implementations."""
import json
import logging
import math
from abc import ABC
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path

import tensorflow as tf

from .player import Player

logger = logging.getLogger("tf_encrypted")


def tensorflow_supports_int64():
    """Test if int64 is supported by this build of TensorFlow. Hacky."""
    x = tf.constant([1], shape=(1, 1), dtype=tf.int64)
    try:
        tf.matmul(x, x)
    except TypeError:
        return False
    return True


def _get_docker_cpu_quota():
    """Checks for available cpu cores in a containerized environment.

    If you witness memory leaks while doing multiple predictions using docker
    see https://github.com/tensorflow/tensorflow/issues/22098.
    """
    cpu_cores = None

    # Check for quotas if we are in a linux container
    cfs_period = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    cfs_quota = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")

    if cfs_period.exists() and cfs_quota.exists():
        with cfs_period.open("rb") as p, cfs_quota.open("rb") as q:
            p_int, q_int = int(p.read()), int(q.read())

            # get the cores allocated by dividing the quota
            # in microseconds by the period in microseconds
            if q_int > 0 and p_int > 0:
                cpu_cores = math.ceil(q_int / p_int)

    return cpu_cores


class Config(ABC):
    """The main tfe.Config abstraction."""

    def __init__(self, debug_mode: bool) -> None:
        self.debug_mode = debug_mode

    @property
    @abstractmethod
    def players(self):
        """Returns the config's list of :class:`Player` objects."""

    @abstractmethod
    def get_player(self, name_or_player):
        """Retrieve a specific :class:`Player` object by name.

        For convenience it is also possible to pass in an existing Player object,
        which will simply be returned as-is if the player is known already.
        """

    @property
    def debug(self) -> bool:
        """
        When debug is True, '@tf.function' decorator will not be used to build graph,
        which makes developers get info easier with python code.
        At the same time, '@memoize' decorator will not cache intermediate results
        to prevent use too much memory.
        """
        return self.debug_mode

    def set_debug_mode(self, debug_mode: bool) -> None:
        self.debug_mode = debug_mode


class LocalConfig(Config):
    """Configure TF Encrypted to use threads on the local CPU.

    Each thread instantiates a different Player to simulate secure computations
    without requiring networking. Mostly intended for development/debugging use.

    By default new players will be added when looked up for the first time;
    this is useful for  instance to get a complete list of players involved
    in a particular computation (see `auto_add_unknown_players`).

    :param (str) player_names: List of players to be used in the session.
    :param str job_name: The name of the job.
    :param bool auto_add_unknown_players: Auto-add player on first lookup.
    """

    def __init__(
        self,
        player_names=None,
        job_name="localhost",
        auto_add_unknown_players=True,
        debug_mode=False,
    ) -> None:
        super(LocalConfig, self).__init__(debug_mode)
        self._job_name = job_name
        self._auto_add_unknown_players = auto_add_unknown_players
        self._players = []
        if player_names is None:
            player_names = []
        for name in player_names:
            self.add_player(name)

    def add_player(self, name):
        index = len(self._players)
        dv_str = "/job:{job_name}/replica:0/task:0/device:CPU:0"
        player = Player(
            name=name,
            index=index,
            device_name=dv_str.format(job_name=self._job_name),
        )
        self._players.append(player)
        return player

    @property
    def players(self):
        return self._players

    def get_player(self, name_or_player):
        if isinstance(name_or_player, Player):
            # we're passed a player
            assert name_or_player in self._players
            return name_or_player

        # we're passed a name
        player = next(
            (player for player in self._players if player.name == name_or_player), None
        )
        if player is None and self._auto_add_unknown_players:
            player = self.add_player(name_or_player)
        return player

    def get_players(self, names):
        if isinstance(names, str):
            names = [name.strip() for name in names.split(",")]
        assert isinstance(names, list)
        return [player for player in self._players if player.name in names]


class RemoteConfig(Config):
    """Configure TF Encrypted to use network hosts for the different players.

    :param (str,str),str->str hostmap: A mapping of hostnames to
        their IP / domain.
    :param str job_name: The name of the job.
    """

    def __init__(self, hostmap, job_name="tfe", debug_mode=False):
        super(RemoteConfig, self).__init__(debug_mode)
        assert isinstance(hostmap, dict)
        if not isinstance(hostmap, OrderedDict):
            logger.warning(
                "Consider passing an ordered dictionary to RemoteConfig instead"
                "in order to preserve host mapping."
            )

        self._job_name = job_name
        device_str = "/job:{job_name}/replica:0/task:{task_id}/device:CPU:0"
        self._players = OrderedDict(
            (
                name,
                Player(
                    name=name,
                    index=index,
                    device_name=device_str.format(job_name=job_name, task_id=index),
                    host=host,
                ),
            )
            for index, (name, host) in enumerate(hostmap.items())
        )

    @staticmethod
    def load(filename):
        """Constructs a RemoteConfig object from a JSON hostmap file.

        :param str filename: Name of file to load from.
        """
        with open(filename, "r") as f:
            hostmap = json.load(f, object_pairs_hook=OrderedDict)
        return RemoteConfig(hostmap)

    def save(self, filename):
        """Saves the configuration as a JSON hostmap file.

        :param str filename: Name of file to save to.
        """
        with open(filename, "w") as f:
            json.dump(self.hostmap, f)

    @property
    def hostmap(self):
        return OrderedDict(
            (player.name, player.host) for player in self._players.values()
        )

    @property
    def hosts(self):
        return [player.host for player in self._players.values()]

    @property
    def players(self):
        return list(self._players.values())

    def get_player(self, name_or_player):
        if isinstance(name_or_player, Player):
            # we're passed a player
            assert name_or_player in self._players.values()
            return name_or_player

        # we're passed a name
        return self._players.get(name_or_player)

    def get_players(self, names):
        if isinstance(names, str):
            names = [name.strip() for name in names.split(",")]
        assert isinstance(names, list)
        return [player for name, player in self._players.items() if name in names]

    def start_server(self, name, start=True):
        """Construct a :class:`tf.distribute.Server` object for the corresponding
        :class:`Player`.

        :param str name: Name of player.
        """
        player = self.get_player(name)
        assert player is not None, "'{}' not found in configuration".format(name)
        cluster = tf.train.ClusterSpec({self._job_name: self.hosts})
        logger.debug("Creating server for '%s' using %s", name, cluster)
        server = tf.distribute.Server(
            cluster, job_name=self._job_name, task_index=player.index, start=start
        )
        logger.info(
            "Created server for '%s' as device '%s'; host address is '%s'",
            name,
            player.device_name,
            server.target,
        )
        return server

    def connect_servers(self):
        cluster = tf.train.ClusterSpec({self._job_name: self.hosts})
        logger.debug("Connect to cluster %s", cluster)
        tf.config.experimental_connect_to_host(self.hosts, job_name=self._job_name)
        logger.debug("Connect to cluster %s succeed", cluster)


__config__ = LocalConfig()


def get_config():
    """Returns the current config."""
    return __config__


def set_config(config) -> None:
    """Sets the current config.

    :param Config config: Intended configuration.
    """
    global __config__
    __config__ = config
