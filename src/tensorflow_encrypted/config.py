import tensorflow as tf
from tensorflow.python.client import timeline

SERVER_0 = '/device:CPU:0'
SERVER_1 = '/device:CPU:1'
CRYPTO_PRODUCER = '/device:CPU:2'
INPUT_PROVIDER  = '/device:CPU:3'
OUTPUT_RECEIVER = '/device:CPU:4'

SESSION_CONFIG = tf.ConfigProto(
    log_device_placement=False,
    device_count={"CPU": 5},
    inter_op_parallelism_threads=1,
    intra_op_parallelism_threads=1
)

TENSORBOARD_DIR = '/tmp/tensorboard'

MASTER = ''

session = lambda: tf.Session(MASTER, config=SESSION_CONFIG)

def run(sess, fetches, tag=None):

    if tag is None:

        sess.run(fetches)

    else:

        writer = tf.summary.FileWriter(TENSORBOARD_DIR, sess.graph)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        sess.run(
            fetches,
            options=run_options,
            run_metadata=run_metadata
        )

        writer.add_run_metadata(run_metadata, tag)
        chrome_trace = timeline.Timeline(run_metadata.step_stats).generate_chrome_trace_format()
        with open('{}/{}.ctr'.format(TENSORBOARD_DIR, tag), 'w') as f:
            f.write(chrome_trace)
