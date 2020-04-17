import contextlib

import tensorflow as tf


class EagerExecutionContext:
    def scope(self):
        return contextlib.suppress()

    def evaluate(self, value):
        return value.numpy()


class GraphExecutionContext:
    def __init__(self):
        self._graph = None
        self._session = None

    @property
    def graph(self):
        if self._graph is None:
            self._graph = tf.Graph()
        return self._graph

    @property
    def session(self):
        if self._session is None:
            with self._graph.as_default():
                self._session = tf.compat.v1.Session()
        return self._session

    def scope(self):
        return self.graph.as_default()

    def evaluate(self, value):
        return self.session.run(value)


def tf_execution_context(run_eagerly):
    if run_eagerly:
        return EagerExecutionContext()
    return GraphExecutionContext()
