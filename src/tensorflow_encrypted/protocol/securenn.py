import numpy as np
from ..protocol import Pond
from ..protocol.pond import _nodes, _lift, _type

class SecureNN(Pond):

    def select_share(self, x, y):
        raise NotImplementedError

    def private_compare(self, x, y):
        raise NotImplementedError

    def share_convert(self,x ,y):
        raise NotImplementedError

    def compute_msb(self, x, y):
        raise NotImplementedError

    def drelu(self, x):
        node_key = ('dReLU', x)
        z = _nodes.get(node_key, None)

        if z is not None:
            return z

        x = _lift(self, x)
        dispatch = {}
        func = dispatch.get(_type(x), None)
        if func is None:
            raise TypeError("Don't know how to dReLU {}".format(type(x)))

        z = func(self, x)
        _nodes[node_key] = z

        return z

    def relu(self, x):
        node_key = ('ReLU', x)
        z = _nodes.get(node_key, None)

        if z is not None:
            return z

        x = _lift(self, x)
        dispatch = {}
        func = dispatch.get(_type(x), None)
        if func is None:
            raise TypeError("Don't know how to ReLU {}".format(type(x)))

        z = func(self, x)
        _nodes[node_key] = z

        return z

    def max_pool(self, x):
        node_key = ('max_pool', x)
        z = _nodes.get(node_key, None)

        if z is not None:
            return z

        x = _lift(self, x)
        dispatch = {}
        func = dispatch.get(_type(x), None)
        if func is None:
            raise TypeError("Don't know how to max_pool {}".format(type(x)))

        z = func(self, x)
        _nodes[node_key] = z

        return z

    def dmax_pool_efficient(self, x):
        node_key = ('dmax_pool', x)
        z = _nodes.get(node_key, None)

        if z is not None:
            return z

        x = _lift(self, x)
        dispatch = {}
        func = dispatch.get(_type(x), None)
        if func is None:
            raise TypeError("Don't know how to dmax_pool {}".format(type(x)))

        z = func(self, x)
        _nodes[node_key] = z

        return z
