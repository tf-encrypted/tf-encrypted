from abc import ABC, abstractmethod, abstractproperty


class Tensor(ABC):

    @property
    @abstractmethod
    def dtype(self):
        pass

    @property
    @abstractmethod
    def shape(self):
        pass

    @property
    @abstractmethod
    def backing(self):
        pass

    def __repr__(self):
        return self.backing.__repr__()
