import torch

from ..utils import tkwargs
from ..base import OptimizationProblem

class Simple(OptimizationProblem):
    def __init__(self, ndim, noise_std=0.0):
        super().__init__(ndim=ndim, nobj=1, noise_std=noise_std)
        self.lb = torch.zeros(self.ndim, **tkwargs)
        self.ub = torch.ones(self.ndim, **tkwargs)

    def _eval(self, x):
        x = torch.atleast_2d(x).to(**tkwargs)
        return torch.sin(x * torch.tensor(2.0) * torch.pi)