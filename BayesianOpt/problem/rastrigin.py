import torch

from ..utils import tkwargs
from ..base import OptimizationProblem

class Rastrigin(OptimizationProblem):
    def __init__(self, ndim):
        super().__init__(ndim=ndim)
        self.lb = -5.12 * torch.ones(self.ndim, **tkwargs)
        self.ub = 5.12 * torch.ones(self.ndim, **tkwargs)

    def eval(self, x):
        x = torch.atleast_2d(x).to(**tkwargs)
        return 10 * self.ndim + torch.sum(x ** 2 - 10 * torch.cos(2 * torch.pi * x), dim=1, keepdim=True)