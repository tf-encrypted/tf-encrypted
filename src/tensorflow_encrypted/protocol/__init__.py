
class Player(object):
    
    def __init__(self, device_name):
        self.device_name = device_name

class Server(Player):
    pass

class CryptoProducer(Player):
    pass

class Protocol(object):
    
    def __enter__(self):
        enter_protocol(self)

    def __exit__(self, exception_type, exception_value, traceback):
        exit_protocol(self)

__active_protocols = []

def get_active_protocol():
    return __active_protocols[-1]

def enter_protocol(protocol):
    assert isinstance(protocol, Protocol)
    __active_protocols.append(protocol)

def exit_protocol(protocol):
    # FIXME[Morten] check that `protocol` is head?
    __active_protocols.pop()

from two_party_spdz import TwoPartySPDZ
from secureml import SecureML
from securenn import SecureNN
