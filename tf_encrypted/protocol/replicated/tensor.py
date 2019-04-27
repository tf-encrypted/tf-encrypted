from abc import ABC, abstractmethod


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

    # def __repr__(self):
    #     return self.backing.__repr__()

    def __add__(self, y):
        return type(self)(self.backing + y.backing, self.dtype)

    def __mul__(self, y):
        return type(self)(self.backing * y.backing, self.dtype)

    def __sub__(self, y):
        return type(self)(self.backing - y.backing, self.dtype)
