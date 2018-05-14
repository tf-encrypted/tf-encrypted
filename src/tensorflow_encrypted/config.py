import tensorflow as tf

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

TENSORBOARD_DIR = '/tmp/tensorflow'

MASTER = ''

session = lambda: tf.Session(MASTER, config=SESSION_CONFIG)
