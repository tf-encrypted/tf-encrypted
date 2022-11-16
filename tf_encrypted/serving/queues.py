# pylint: disable=missing-docstring

import numpy as np
import tensorflow as tf

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
        input_queue_capacity=10,
        input_queue_name="input",
        output_queue_capacity=10,
        output_queue_name="output",
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.input_queue = tfe.queue.FIFOQueue(
            capacity=input_queue_capacity,
            shape=input_shape,
            shared_name=input_queue_name,
        )

        self.output_queue = tfe.queue.FIFOQueue(
            capacity=output_queue_capacity,
            shape=output_shape,
            shared_name=output_queue_name,
        )

        # computation step
        @tf.function
        def computation_step():
            x = self.input_queue.dequeue()
            y = computation_fn(x)
            self.output_queue.enqueue(y)

        self.run_step = computation_step

    def run(self, num_steps=None, step_fn=None):
        """
        Continuously serve computations, for `num_steps` if specified.

        If specified, `step_fn` is called after each step.
        """
        if num_steps is not None:
            for _ in range(num_steps):
                self.run_step()
                if step_fn is not None:
                    step_fn()
        else:
            while True:
                self.run_step()
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
        input_queue_capacity=10,
        input_queue_name="input",
        output_queue_capacity=10,
        output_queue_name="output",
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape

        # input
        self.input_queue = tfe.queue.FIFOQueue(
            capacity=input_queue_capacity,
            shape=input_shape,
            shared_name=input_queue_name,
        )

        # output
        self.output_queue = tfe.queue.FIFOQueue(
            capacity=output_queue_capacity,
            shape=output_shape,
            shared_name=output_queue_name,
        )

    def send_input(self, x):
        """
        Send `x` to servers for processing.
        """
        assert isinstance(x, np.ndarray)
        assert list(x.shape) == list(self.input_shape)

        self.input_queue.enqueue(tfe.define_private_variable(x))

    def receive_output(self):
        """
        Receive result from servers, blocking until available.
        """
        return self.output_queue.dequeue().reveal().to_native()

    def run(self, x):
        """
        Send `x` to servers and return result.
        """
        self.send_input(x)
        return self.receive_output()
