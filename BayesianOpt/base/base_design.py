from abc import ABC, abstractmethod

class ExperimentalDesgin(ABC):
    def __init__(self, ndim, lb, ub, random_state):
        self.ndim = ndim
        self.lb = lb
        self.ub = ub
        self.random_state = random_state

    @abstractmethod
    def generate_points(self, npts):
        raise NotImplementedError