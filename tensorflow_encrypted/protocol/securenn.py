from ..protocol import Pond


class SecureNN(Pond):

    def select_share(self, x, y):
        raise NotImplementedError

    def bitwise_xor(self, x, y):
        raise NotImplementedError

    def private_compare(self, x, r, beta):
        raise NotImplementedError

    def share_convert(self, x):
        raise NotImplementedError

    def compute_msb(self, x):
        raise NotImplementedError

    def divide(self, x, y):
        raise NotImplementedError

    def drelu(self, x):
        raise NotImplementedError

    def relu(self, x):
        raise NotImplementedError

    def max_pool(self, x):
        raise NotImplementedError

    def dmax_pool_efficient(self, x):
        raise NotImplementedError
