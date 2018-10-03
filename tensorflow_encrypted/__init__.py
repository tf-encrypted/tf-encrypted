from __future__ import absolute_import
import tensorflow as tf

from .config import LocalConfig, RemoteConfig, get_default_config, Session
from .session import setTFEDebugFlag, setMonitorStatsFlag
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
    "get_default_config",
    "Session",
    "io",
    "player",
    "protocol",
    "layers",
    "convert",
    "global_caches_updator",
]
