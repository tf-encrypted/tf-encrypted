from types import Dict
from collections import defaultdict

import tensorflow as tf
from tensorflow.python.client import timeline

TENSORBOARD_DIR = '/tmp/tensorboard'
IGNORE_STATS = False
DEBUG = True

_run_counter = defaultdict(int)

def local_session(num_players:int, log_device_placement:bool=False) -> tf.Session:
    """
    Creates a session using threads on the local CPU to simulate the different players.
    Intended mostly for development/debugging use.
    """
    
    return tf.Session(
        '',
        config=tf.ConfigProto(
            log_device_placement=log_device_placement,
            allow_soft_placement=False,
            device_count={"CPU": num_players},
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1
        )
    )

def remote_session(master_host:str, log_device_placement:bool=False) -> tf.Session:

    master_uri = 'grpc://{}'.format(master_host)

    CONFIG = tf.ConfigProto(
        log_device_placement=log_device_placement,
        allow_soft_placement=False,
    )

    return tf.Session(
        master_uri,
        config=CONFIG
    )

def run(sess: tf.Session, fetches, feed_dict:Dict={}, tag=None) -> None:

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
