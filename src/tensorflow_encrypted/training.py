
import tensorflow as tf
from ops import *
from ops import m

def define_batch_buffer(shape, capacity, varname):

    # each server holds three values: xi, ai, and alpha
    server_packed_shape = (3, len(m)) + tuple(shape)

    # the crypto producer holds just one value: a
    cryptoprovider_packed_shape = (len(m),) + tuple(shape)

    with tf.device(SERVER_0):
        queue_0 = tf.FIFOQueue(
            capacity=capacity,
            dtypes=[INT_TYPE],
            shapes=[server_packed_shape],
            name='buffer_{}_0'.format(varname),
        )

    with tf.device(SERVER_1):
        queue_1 = tf.FIFOQueue(
            capacity=capacity,
            dtypes=[INT_TYPE],
            shapes=[server_packed_shape],
            name='buffer_{}_1'.format(varname),
        )

    with tf.device(CRYPTO_PRODUCER):
        queue_cp = tf.FIFOQueue(
            capacity=capacity,
            dtypes=[INT_TYPE],
            shapes=[cryptoprovider_packed_shape],
            name='buffer_{}_cp'.format(varname),
        )

    return (queue_0, queue_1, queue_cp)

def pack_server(tensors):
    with tf.name_scope('pack'):
        return tf.stack([ tf.stack(tensor, axis=0) for tensor in tensors ], axis=0)

def unpack_server(shape, tensors):
    with tf.name_scope('unpack'):
        return [
            [
                tf.reshape(subtensor, shape)
                for subtensor in tf.split(tf.reshape(tensor, (len(m),) + shape), len(m))
            ]
            for tensor in tf.split(tensors, 3)
        ]

def pack_cryptoproducer(tensor):
    with tf.name_scope('pack'):
        return tf.stack(tensor, axis=0)

def unpack_cryptoproducer(shape, tensor):
    with tf.name_scope('unpack'):
        return [
            tf.reshape(subtensor, shape)
            for subtensor in tf.split(tensor, len(m))
        ]

def distribute_batch(shape, buffer, varname):

    queue_0, queue_1, queue_cp = buffer

    with tf.name_scope('distribute_{}'.format(varname)):

        input_x = [ tf.placeholder(INT_TYPE, shape=shape) for _ in m ]

        with tf.device(INPUT_PROVIDER):
            with tf.name_scope('preprocess'):
                # share x
                x0, x1 = share(input_x)
                # precompute mask
                a = sample(shape)
                a0, a1 = share(a)
                alpha = crt_sub(input_x, a)

        with tf.device(SERVER_0):
            enqueue_0 = queue_0.enqueue(pack_server([x0, a0, alpha]))

        with tf.device(SERVER_1):
            enqueue_1 = queue_1.enqueue(pack_server([x1, a1, alpha]))

        with tf.device(CRYPTO_PRODUCER):
            enqueue_cp = queue_cp.enqueue(pack_cryptoproducer(a))

        populate_op = [enqueue_0, enqueue_1, enqueue_cp]

    return input_x, populate_op

def load_batch(shape, buffer, varname):

    shape = tuple(shape)
    queue_0, queue_1, queue_cp = buffer

    with tf.name_scope('load_batch_{}'.format(varname)):

        with tf.device(SERVER_0):
            packed_0 = queue_0.dequeue()
            x0, a0, alpha_on_0 = unpack_server(shape, packed_0)

        with tf.device(SERVER_1):
            packed_1 = queue_1.dequeue()
            x1, a1, alpha_on_1 = unpack_server(shape, packed_1)

        with tf.device(CRYPTO_PRODUCER):
            packed_cp = queue_cp.dequeue()
            a = unpack_cryptoproducer(shape, packed_cp)

    x = PrivateTensor(x0, x1)
    
    node_key = ('mask', x)
    nodes[node_key] = (a, a0, a1, alpha_on_0, alpha_on_1)

    return x

def training_loop(buffers, shapes, iterations, initial_weights, training_step):

    buffer_x, buffer_y = buffers
    shape_x, shape_y = shapes

    initial_w0, initial_w1 = share(initial_weights)

    def loop_op(w0, w1):
        w = PrivateTensor(w0, w1)
        x = load_batch(shape_x, buffer_x, varname='x')
        y = load_batch(shape_y, buffer_y, varname='y')
        new_w = training_step(w, x, y)
        return new_w.share0, new_w.share1

    _, final_w0, final_w1 = tf.while_loop(
        cond=lambda i, w0, w1: tf.less(i, iterations),
        body=lambda i, w0, w1: (i+1,) + loop_op(w0, w1),
        loop_vars=(0, initial_w0, initial_w1),
        parallel_iterations=1
    )

    return final_w0, final_w1

buffer_x = define_batch_buffer(shape_x, num_batches, varname='x')
buffer_y = define_batch_buffer(shape_y, num_batches, varname='y')

input_x, distribute_x = distribute_batch(shape_x, buffer_x, varname='x')
input_y, distribute_y = distribute_batch(shape_y, buffer_y, varname='y')