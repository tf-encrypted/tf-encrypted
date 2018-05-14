
from ..protocol import Protocol

class TwoPartySPDZ(Protocol):

    def __init__(self, server0, server1, crypto_producer):
        self.server0 = server0
        self.server1 = server1
        self.crypto_producer = crypto_producer
