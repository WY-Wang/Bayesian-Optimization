from abc import ABC, abstractmethod

import torch


class OptimizationProblem(ABC):
    def __init__(self, ndim, nobj, noise_std):
        self.ndim = ndim
        self.nobj = nobj
        self.noise_std = noise_std

        self._name = self.__class__.__name__
        self.name = self._name + f"({self.noise_std})"

    def eval(self, x):
        fx = self._eval(x)
        return fx.add(torch.randn_like(fx).mul(self.noise_std))

    @abstractmethod
    def _eval(self, x):
        raise NotImplementedError