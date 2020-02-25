"""
FIFO queue data structure.
"""

from __future__ import absolute_import
import abc

import tf_encrypted as tfe


class AbstractFIFOQueue(abc.ABC):
    """
  FIFO queues mimicking `tf.queue.FIFOQueue`.
  """

    @abc.abstractmethod
    def enqueue(self, tensor):
        """
    Push `tensor` onto queue.

    Blocks if queue is full.
    """

    @abc.abstractmethod
    def dequeue(self):
        """
    Pop tensor from queue.

    Blocks if queue is empty.
    """


def FIFOQueue(capacity, shape, shared_name=None):
    return tfe.fifo_queue(
        capacity=capacity,
        shape=shape,
        shared_name=shared_name,
    )
