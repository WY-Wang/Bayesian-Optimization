from abc import ABC, abstractmethod

class OptimizationProblem(ABC):
    def __init__(self, ndim, nobj = 1):
        self.ndim = ndim
        self.nobj = nobj
        self.name = self.__class__.__name__

    @abstractmethod
    def eval(self, x):
        raise NotImplementedError