from __future__ import absolute_import
import tensorflow as tf

from .config import LocalConfig, RemoteConfig, get_config, set_config
from .session import setTFEDebugFlag, setMonitorStatsFlag, Session
from .protocol import global_caches_updator
from .player import player
from . import io
from . import protocol
from . import layers
from . import convert


def set_random_seed(seed: int) -> None:
    tf.set_random_seed(seed)


__all__ = [
    "LocalConfig",
    "RemoteConfig",
    "setTFEDebugFlag",
    "setMonitorStatsFlag",
    "get_config",
    "set_config",
    "Session",
    "io",
    "player",
    "protocol",
    "layers",
    "convert",
    "global_caches_updator",
]
