from typing import Dict, List, Optional, Any, Union, Tuple, NamedTuple
from collections import defaultdict

import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline

from .protocol import Player


class LocalConfig(object):
    """
    Configure tf-encrypted to use threads on the local CPU
    to simulate the different players.
    Intended mostly for development/debugging use.
    """

    def __init__(self, num_players: int, job_name: str='localhost') -> None:
        self.num_players = num_players
        self.job_name = job_name

    @property
    def players(self) -> List[Player]:
        return [
            Player(
                '/job:{job_name}/replica:0/task:0/device:CPU:{cpu_id}'.format(
                    job_name=self.job_name,
                    # swift by one to allow for master to be `CPU:0`
                    cpu_id=index+1
                )
            )
            for index in range(self.num_players)
        ]

    def session(self, log_device_placement: bool=False) -> tf.Session:
        # reserve one CPU for the player executing the script, to avoid
        # default pinning of operations to one of the actual players
        return tf.Session(
            '',
            None,
            config=tf.ConfigProto(
                log_device_placement=log_device_placement,
                allow_soft_placement=False,
                device_count={"CPU": self.num_players + 1},
                inter_op_parallelism_threads=1,
                intra_op_parallelism_threads=1
            )
        )


class RemoteConfig(object):
    """
    Configure tf-encrypted to use network hosts for the different players.
    """

    def __init__(self, player_hosts: List[str],
                 master_host: Optional[str]=None,
                 job_name: str='tfe') -> None:

        self.player_hosts = player_hosts
        self.master_host = master_host
        self.job_name = job_name

    @property
    def players(self) -> List[Player]:
        if self.master_host is None:
            offset = 0
        else:
            offset = 1

        return [
            Player('/job:{job_name}/replica:0/task:{task_id}/cpu:0'.format(
                job_name=self.job_name,
                task_id=index+offset
            ))
            for index in range(len(self.player_hosts))
        ]

    def server(self, player_index: int) -> tf.train.Server:
        if self.master_host is None:
            # use first player as master so don't add any
            cluster = tf.train.ClusterSpec({self.job_name: self.player_hosts})
            return tf.train.Server(cluster, job_name=self.job_name,
                                   task_index=player_index)
        else:
            # add explicit master to cluster
            cluster = tf.train.ClusterSpec(
                {self.job_name: [self.master_host] + self.player_hosts}
            )
            return tf.train.Server(cluster, job_name=self.job_name,
                                   task_index=player_index+1)

    def session(self, log_device_placement: bool=False) -> tf.Session:
        if self.master_host is None:
            # use first player as master
            target = 'grpc://{}'.format(self.player_hosts[0])
        else:
            # use specified master
            target = 'grpc://{}'.format(self.master_host)

        config = tf.ConfigProto(
            log_device_placement=log_device_placement,
            allow_soft_placement=False,
        )

        return tf.Session(
            target,
            config=config
        )


TENSORBOARD_DIR = '/tmp/tensorboard'
IGNORE_STATS = False
DEBUG = True

_run_counter: Any = defaultdict(int)


def run(
    sess: tf.Session,
    fetches: Union[tf.GraphElement,
                   List[tf.GraphElement],
                   Tuple[tf.GraphElement],
                   NamedTuple,
                   Dict[str, tf.GraphElement]],
    feed_dict: Dict[str, np.ndarray]={},
    tag: Optional[str] = None
) -> Any:

    if not DEBUG and (tag is None or IGNORE_STATS):

        return sess.run(
            fetches,
            feed_dict=feed_dict
        )

    else:

        session_tag = '{}{}'.format(tag, _run_counter[tag])
        # run_tag = TENSORBOARD_DIR + ('/' + tag if tag is not None else '')
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
