"""Various sources for providing generalized Beaver triples for the Pond
protocol."""
import abc
import logging
import random

import tensorflow as tf

from ...config import get_config
from ...utils import wrap_in_variables, reachable_nodes, unwrap_fetches


logging.basicConfig()
logger = logging.getLogger('tf_encrypted')
logger.setLevel(logging.DEBUG)


class TripleSource(abc.ABC):
  """Base class for triples sources."""

  @abc.abstractmethod
  def cache(self, a, cache_updater):
    pass

  @abc.abstractmethod
  def initializer(self):
    pass

  @abc.abstractmethod
  def generate_triples(self, fetches):
    pass


class BaseTripleSource(TripleSource):
  """
  Partial triple source adding graph nodes for constructing and keeping track
  of triples and their use. Subclasses must implement `_build_queues`.
  """

  def __init__(self, player0, player1, producer):
    config = get_config()
    self.player0 = config.get_player(player0) if player0 else None
    self.player1 = config.get_player(player1) if player1 else None
    self.producer = config.get_player(producer) if producer else None

  def mask(self, backing_dtype, shape):

    with tf.name_scope("triple-generation"):
      with tf.device(self.producer.device_name):
        a0 = backing_dtype.sample_uniform(shape)
        a1 = backing_dtype.sample_uniform(shape)
        a = a0 + a1

    d0, d1 = self._build_queues(a0, a1)
    return a, d0, d1

  def mul_triple(self, a, b):

    with tf.name_scope("triple-generation"):
      with tf.device(self.producer.device_name):
        ab = a * b
        ab0, ab1 = self._share(ab)

    return self._build_queues(ab0, ab1)

  def square_triple(self, a):

    with tf.name_scope("triple-generation"):
      with tf.device(self.producer.device_name):
        aa = a * a
        aa0, aa1 = self._share(aa)

    return self._build_queues(aa0, aa1)

  def matmul_triple(self, a, b):

    with tf.name_scope("triple-generation"):
      with tf.device(self.producer.device_name):
        ab = a.matmul(b)
        ab0, ab1 = self._share(ab)

    return self._build_queues(ab0, ab1)

  def conv2d_triple(self, a, b, strides, padding):

    with tf.device(self.producer.device_name):
      with tf.name_scope("triple"):
        ab = a.conv2d(b, strides, padding)
        ab0, ab1 = self._share(ab)

    return self._build_queues(ab0, ab1)

  def indexer_mask(self, a, slc):

    with tf.name_scope("mask-transformation"):
      with tf.device(self.producer.device_name):
        a_sliced = a[slc]

    return a_sliced

  def transpose_mask(self, a, perm):

    with tf.name_scope("mask-transformation"):
      with tf.device(self.producer.device_name):
        a_t = a.transpose(perm=perm)

    return a_t

  def strided_slice_mask(self, a, args, kwargs):

    with tf.name_scope("mask-transformation"):
      with tf.device(self.producer.device_name):
        a_slice = a.strided_slice(args, kwargs)

    return a_slice

  def split_mask(self, a, num_split, axis):

    with tf.name_scope("mask-transformation"):
      with tf.device(self.producer.device_name):
        bs = a.split(num_split=num_split, axis=axis)

    return bs

  def stack_mask(self, bs, axis):

    factory = bs[0].factory

    with tf.name_scope("mask-transformation"):
      with tf.device(self.producer.device_name):
        b_stacked = factory.stack(bs, axis=axis)

    return b_stacked

  def concat_mask(self, bs, axis):

    factory = bs[0].factory

    with tf.name_scope("mask-transformation"):
      with tf.device(self.producer.device_name):
        b_stacked = factory.concat(bs, axis=axis)

    return b_stacked

  def reshape_mask(self, a, shape):

    with tf.name_scope("mask-transformation"):
      with tf.device(self.producer.device_name):
        a_reshaped = a.reshape(shape)

    return a_reshaped

  def expand_dims_mask(self, a, axis):

    with tf.name_scope("mask-transformation"):
      with tf.device(self.producer.device_name):
        a_e = a.expand_dims(axis=axis)

    return a_e

  def squeeze_mask(self, a, axis):

    with tf.name_scope("mask-transformation"):
      with tf.device(self.producer.device_name):
        a_squeezed = a.squeeze(axis=axis)

    return a_squeezed

  def _share(self, secret):
    with tf.name_scope("share"):
      share0 = secret.factory.sample_uniform(secret.shape)
      share1 = secret - share0
      # randomized swap to distribute who gets the seed
      if random.random() < 0.5:
        share0, share1 = share1, share0
    return share0, share1

  @abc.abstractmethod
  def _build_queues(self, c0, c1):
    """
    Method used to inject buffers between mask generating and use
    (ie online vs offline). `c0` and `c1` represent the generated
    masks and the method is expected to return a similar pair of
    of tensors.
    """


class OnlineTripleSource(BaseTripleSource):
  """
  This triple source will generate triples as part of the online phase
  using a dedicated third-party `producer`.

  There is no need to call `generate_triples` nor `initialize`.
  """

  def __init__(self, producer):
    super().__init__(None, None, producer)

  def cache(self, a, cache_updater):
    with tf.device(self.producer.device_name):
      updater, [a_cached] = wrap_in_variables(a)
    return updater, a_cached

  def initializer(self):
    return tf.no_op()

  def generate_triples(self, fetches):
    return []

  def _build_queues(self, c0, c1):
    return c0, c1


class QueuedOnlineTripleSource(BaseTripleSource):
  """
  Similar to `OnlineTripleSource` but with in-memory buffering backed by
  `tf.FIFOQueue`s.
  """

  def __init__(self, player0, player1, producer, capacity=10):
    super().__init__(player0, player1, producer)
    self.capacity = capacity
    self.queues = list()
    self.triggers = dict()

  def cache(self, a, cache_updater):
    with tf.device(self.producer.device_name):
      offline_updater, [a_cached] = wrap_in_variables(a)
    self.triggers[cache_updater] = offline_updater
    return tf.no_op(), a_cached

  def initializer(self):
    return tf.no_op()

  def generate_triples(self, fetches):
    if isinstance(fetches, (list, tuple)) and len(fetches) > 1:
      logger.warning("Generating triples for a run involving more than "
                     "one fetch may introduce non-determinism that can "
                     "break the correspondence between the two phases "
                     "of the computation.")

    unwrapped_fetches = unwrap_fetches(fetches)
    reachable_operations = [node
                            for node in reachable_nodes(unwrapped_fetches)
                            if isinstance(node, tf.Operation)]
    reachable_triggers = [self.triggers[op]
                          for op in reachable_operations
                          if op in self.triggers]
    return reachable_triggers

  def _build_triple_store(self, mask, player_id):
    """
    Adds a tf.FIFOQueue to store mask locally on player.
    """

    # TODO(Morten) taking `value` doesn't work for int100
    raw_mask = mask.value
    factory = mask.factory
    dtype = mask.factory.native_type
    shape = mask.shape

    with tf.name_scope("triple-store-{}".format(player_id)):

      q = tf.queue.FIFOQueue(
          capacity=self.capacity,
          dtypes=[dtype],
          shapes=[shape],
      )
      e = q.enqueue(raw_mask)
      d = q.dequeue()
      d_wrapped = factory.tensor(d)

    self.queues += [q]
    self.triggers[d.op] = e
    return d_wrapped

  def _build_queues(self, c0, c1):

    with tf.device(self.player0.device_name):
      d0 = self._build_triple_store(c0, "0")

    with tf.device(self.player1.device_name):
      d1 = self._build_triple_store(c1, "1")

    return d0, d1


"""
class PlaceholderTripleSource(BaseTripleSource):

    # TODO(Morten) manually unwrap and re-wrap of values, should be hidden away

    def __init__(self, player0, player1, producer):
        super().__init__(player0, player1, producer)
        self.placeholders = list()

    def _build_queues(self, c0, c1):

        with tf.device(self.player0.device_name):
            r0 = tf.placeholder(
                dtype=c0.factory.native_type,
                shape=c0.shape,
            )
            d0 = c0.factory.tensor(r0)

        with tf.device(self.player1.device_name):
            r1 = tf.placeholder(
                dtype=c1.factory.native_type,
                shape=c1.shape,
            )
            d1 = c1.factory.tensor(r1)

        self.placeholders += [r0, r1]
        return d0, d1
"""  #pylint: disable=pointless-string-statement


"""
class DatasetTripleSource(BaseTripleSource):

    # TODO(Morten) manually unwrap and re-wrap of values, should be hidden away

    def __init__(
        self,
        player0,
        player1,
        producer,
        capacity=10,
        directory="/tmp/triples/",
        support_online_running=False,
    ):
        super().__init__(player0, player1, producer)
        self.capacity = capacity
        self.dequeuers = list()
        self.enqueuers = list()
        self.initializers = list()
        self.directory = directory
        self.support_online_running = support_online_running
        if support_online_running:
            self.dequeue_from_file = tf.placeholder_with_default(True,
                                                                 shape=[])

    def _build_queues(self, c0, c1):

        def dataset_from_file(filename, dtype, shape):
            def parse(x):
                res = tf.parse_tensor(x, out_type=dtype)
                res = tf.reshape(res, shape)
                return res
            iterator = tf.data.TFRecordDataset(filename) \
                .map(parse) \
                .make_initializable_iterator()
            return iterator.get_next(), iterator.initializer

        def dataset_from_queue(queue, dtype, shape):
            dummy = tf.data.Dataset.from_tensors(0).repeat(None)
            iterator = (dummy.map(lambda _: queue.dequeue())
                             .make_initializable_iterator())
            return iterator.get_next(), iterator.initializer
            # gen = lambda: queue.dequeue()
            # dataset = tf.data.Dataset.from_generator(gen, [dtype], [shape])
            # iterator = dataset.make_one_shot_iterator()
            # return iterator.get_next(), iterator.initializer

        def sanitize_filename(filename):
            return filename.replace('/', '__')

        def build_triple_store(mask):

            raw_mask = mask.value
            factory = mask.factory
            dtype = mask.factory.native_type
            shape = mask.shape

            with tf.name_scope("triple-store"):

                q = tf.queue.FIFOQueue(
                    capacity=self.capacity,
                    dtypes=[dtype],
                    shapes=[shape],
                )

                e = q.enqueue(raw_mask)
                f = os.path.join(self.directory, sanitize_filename(q.name))

                if self.support_online_running:
                    r, i = tf.cond(
                        self.dequeue_from_file,
                        true_fn=lambda: dataset_from_file(f, dtype, shape),
                        false_fn=lambda: dataset_from_queue(q, dtype, shape),
                    )
                else:
                    r, i = dataset_from_file(f, dtype, shape)
                d = factory.tensor(r)

            return f, q, e, d, i

        with tf.device(self.player0.device_name):
            f0, q0, e0, d0, i0 = build_triple_store(c0)

        with tf.device(self.player1.device_name):
            f1, q1, e1, d1, i1 = build_triple_store(c1)

        self.dequeuers += [(f0, q0.dequeue()), (f1, q1.dequeue())]
        self.enqueuers += [(e0, e1)]
        self.initializers += [(i0, i1)]
        return d0, d1

    def initialize(self, sess, tag=None):
        sess.run(self.initializers, tag=tag)

    def generate_triples(self, sess, num=1, tag=None, save_to_file=True):
        for _ in range(num):
            sess.run(self.enqueuers, tag=tag)

        if save_to_file:
            self.save_triples_to_file(sess, num=num, tag=tag)

    def save_triples_to_file(self, sess, num, tag=None):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        for filename, dequeue in self.dequeuers:
            with tf.io.TFRecordWriter(filename) as writer:
                # size = sess.run(queue.size(), tag=tag)
                for _ in range(num):
                    serialized = tf.io.serialize_tensor(dequeue)
                    triple = sess.run(serialized, tag=tag)
                    writer.write(triple)
"""  #pylint: disable=pointless-string-statement
