import torch

from ..utils import tkwargs
from ..base import OptimizationProblem

class Ackley(OptimizationProblem):
    def __init__(self, ndim):
        super().__init__(ndim=ndim)
        self.lb = -15 * torch.ones(self.ndim, **tkwargs)
        self.ub = 20 * torch.ones(self.ndim, **tkwargs)

    def eval(self, x):
        x = torch.atleast_2d(x).to(**tkwargs)
        return (
            -20.0 * torch.exp(-0.2 * torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) / self.ndim))
            - torch.exp(torch.sum(torch.cos(2.0 * torch.pi * x), dim=1, keepdim=True) / self.ndim)
            + 20
            + torch.exp(torch.tensor(1.0))
        )