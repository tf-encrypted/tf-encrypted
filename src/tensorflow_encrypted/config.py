from collections import defaultdict

import tensorflow as tf
from tensorflow.python.client import timeline

TENSORBOARD_DIR = '/tmp/tensorboard'
IGNORE_STATS = False

_run_counter = defaultdict(int)

def session(num_players):
    return tf.Session(
        '',
        config=tf.ConfigProto(
            log_device_placement=False,
            device_count={"CPU": num_players},
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1
        )
    )

def run(sess, fetches, tag=None):

    if tag is None or IGNORE_STATS:

        return sess.run(fetches)

    else:

        run_tag = TENSORBOARD_DIR + ('/' + tag if tag is not None else '')        
        session_tag = '{}{}'.format(tag, _run_counter[tag])
        _run_counter[tag] += 1

        writer = tf.summary.FileWriter(run_tag, sess.graph)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        results = sess.run(
            fetches,
            options=run_options,
            run_metadata=run_metadata
        )

        writer.add_run_metadata(run_metadata, session_tag)
        chrome_trace = timeline.Timeline(run_metadata.step_stats).generate_chrome_trace_format()
        with open('{}/{}.ctr'.format(TENSORBOARD_DIR, session_tag), 'w') as f:
            f.write(chrome_trace)

        return results
