"""
TODO
"""
import numpy as np

import tf_encrypted as tfe

class QueueServer:
  """
  Serving server based on `tfe.queue.FIFOQueue`.
  """

  def __init__(
      self,
      input_shape,
      output_shape,
      computation_fn,
      input_queue_capacity=1,
      input_queue_name="input",
      output_queue_capacity=1,
      output_queue_name="output",
  ):
    self.input_shape = input_shape
    self.output_shape = output_shape

    # input
    self.input_queue = tfe.queue.FIFOQueue(
        capacity=input_queue_capacity,
        shape=input_shape,
        shared_name=input_queue_name)

    # output
    self.output_queue = tfe.queue.FIFOQueue(
        capacity=output_queue_capacity,
        shape=output_shape,
        shared_name=output_queue_name)

    # computation step
    x = self.input_queue.dequeue()
    y = computation_fn(x)
    self.step_op = self.output_queue.enqueue(y)

  def run_step(self, sess, tag='step'):
    """
    Serve single computation.
    """
    sess.run(self.step_op, tag=tag)

  def run(self, sess, num_steps=None, step_fn=None):
    """
    Continuously serve computations, for `num_steps` if specified.

    If specified, `step_fn` is called after each step.
    """
    if num_steps is not None:
      for _ in range(num_steps):
        self.run_step(sess)
        if step_fn is not None:
          step_fn()
    else:
      while True:
        self.run_step(sess)
        if step_fn is not None:
          step_fn()


class QueueClient:
  """
  Serving client based on `tfe.queue.FIFOQueue`.

  Must be set up with the same arguments as the server.
  """

  def __init__(
      self,
      input_shape,
      output_shape,
      input_queue_capacity=1,
      input_queue_name="input",
      output_queue_capacity=1,
      output_queue_name="output",
  ):
    self.input_shape = input_shape
    self.output_shape = output_shape

    # input
    input_queue = tfe.queue.FIFOQueue(
        capacity=input_queue_capacity,
        shape=input_shape,
        shared_name=input_queue_name)
    self.input_placeholder = tfe.define_private_placeholder(shape=input_shape)
    self.input_op = input_queue.enqueue(self.input_placeholder)

    # output
    output_queue = tfe.queue.FIFOQueue(
        capacity=output_queue_capacity,
        shape=output_shape,
        shared_name=output_queue_name)
    output = output_queue.dequeue()
    self.output0 = output.share0
    self.output1 = output.share1

    # fixedpoint config
    self.modulus = output.backing_dtype.modulus
    self.bound = tfe.get_protocol().fixedpoint_config.bound_single_precision
    self.scaling_factor = tfe.get_protocol().fixedpoint_config.scaling_factor

  def send_input(self, sess, x, tag='input'):
    """
    Send `x` to servers for processing.
    """
    assert isinstance(x, np.ndarray)
    assert list(x.shape) == list(self.input_shape)

    # simply run the input op with
    sess.run(
        self.input_op,
        tag=tag,
        output_partition_graphs=True,
        feed_dict=self.input_placeholder.feed(x),
    )

  def receive_output(self, sess, tag='output'):
    """
    Receive result from servers, blocking until available.
    """
    res0, res1 = sess.run(
        [self.output0, self.output1],
        tag=tag,
        output_partition_graphs=True,
    )

    res = (res0 + res1) % self.modulus
    res = (res + self.bound) % self.modulus - self.bound
    res = res / self.scaling_factor
    return res

  def run(self, sess, x):
    """
    Send `x` to servers and return result.
    """
    self.send_input(sess, x)
    return self.receive_output(sess)
