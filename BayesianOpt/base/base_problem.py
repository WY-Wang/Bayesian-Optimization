from abc import ABC, abstractmethod

import torch


class OptimizationProblem(ABC):
    def __init__(self, ndim, nobj = 1):
        self.ndim = ndim
        self.nobj = nobj
        self.name = self.__class__.__name__

    def eval_noisy(self, x, mean=0.0, std=1.0):
        fx = self.eval(x)
        return fx.add(torch.randn_like(fx).mul(std).add(mean))

    @abstractmethod
    def eval(self, x):
        raise NotImplementedError