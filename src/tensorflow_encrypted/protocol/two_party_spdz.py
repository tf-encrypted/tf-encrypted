
from ..protocol import Protocol

class TwoPartySPDZ(Protocol):

    def __init__(self, server_0, server_1, crypto_producer):
        self.server_0 = server_0
        self.server_1 = server_1
        self.crypto_producer = crypto_producer
