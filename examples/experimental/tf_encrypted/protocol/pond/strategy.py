import tensorflow as tf


class Pond(tf.distribute.Strategy):
  def __init__(self, triples_source, *players, **kwargs):
    extended = PondExtended(self, triples_source, *players, **kwargs)
    super.__init__(extended)
    # TODO

  def reveal(self, *args):
    for x in args:
      self.reduce(reduce_op=tf.distribute.ReduceOp.SUM, value=x, axis=None)

class PondExtended(tf.distribute.StrategyExtended):
  def __init__(self, container_strategy, triples_source, *players, **kwargs):
    super().__init__(container_strategy)
    # TODO
