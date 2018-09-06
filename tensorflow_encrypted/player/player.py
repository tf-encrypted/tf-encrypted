from typing import Optional


class Player(object):
    def __init__(self, name: str, index: int, device_name: str, host: Optional[str] = None) -> None:
        self.name = name
        self.index = index
        self.device_name = device_name
        self.host = host
