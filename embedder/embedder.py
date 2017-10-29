from abc import ABCMeta, abstractmethod


class Embedder(metaclass=ABCMeta):
    @abstractmethod
    def embed(self, x):
        raise NotImplementedError()
