
class Player(object):
    
    def __init__(self, device_name):
        self.device_name = device_name

class Server(Player):
    pass

class CryptoProducer(Player):
    pass

class Protocol(object):
    pass

# from unencrypted_native import UnencryptedNative
# from unencrypted_fixedpoint import UnencryptedFixedpoint
from pond import Pond
# from secureml import SecureML
# from securenn import SecureNN
