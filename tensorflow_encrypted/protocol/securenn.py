from .protocol import cached
from ..protocol.pond import (
    Pond, PondTensor
)


class SecureNN(Pond):

    @cached
    def bitwise_not(self, x: PondTensor) -> PondTensor:
        assert not x.is_scaled, "Input is not supposed to be scaled"
        return self.sub(1, x)

    @cached
    def bitwise_and(self, x: PondTensor, y: PondTensor) -> PondTensor:
        assert (not x.is_scaled) and (not y.is_scaled), "Inputs are not supposed to be scaled"
        return self.mul(x, y)

    @cached
    def bitwise_or(self, x: PondTensor, y: PondTensor) -> PondTensor:
        assert (not x.is_scaled) and (not y.is_scaled), "Inputs are not supposed to be scaled"
        return x + y - self.bitwise_and(x, y)

    @cached
    def bitwise_xor(self, x: PondTensor, y: PondTensor) -> PondTensor:
        assert (not x.is_scaled) and (not y.is_scaled), "Inputs are not supposed to be scaled"
        return x + y - self.bitwise_and(x, y) * 2

    @cached
    def msb(self, x: PondTensor) -> PondTensor:
        # NOTE when the modulus is odd then msb reduces to lsb via x -> 2*x
        # TODO assert that we're actually using an odd modulus
        return self.lsb(x * 2)

    def lsb(self, x: PondTensor) -> PondTensor:
        raise NotImplementedError

    def select_share(self, x, y):
        raise NotImplementedError

    def private_compare(self, x, r, beta):
        raise NotImplementedError

    def share_convert(self, x):
        raise NotImplementedError

    def divide(self, x, y):
        raise NotImplementedError

    def drelu(self, x):
        raise NotImplementedError

    @cached
    def relu(self, x):
        return self.drelu(x) * x

    def max_pool(self, x):
        raise NotImplementedError

    def dmax_pool_efficient(self, x):
        raise NotImplementedError
