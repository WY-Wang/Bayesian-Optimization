import torch

from ..utils import tkwargs
from ..base import OptimizationProblem


class Branin(OptimizationProblem):
    def __init__(self, noise_std=0.0, **params):
        super().__init__(ndim=2, nobj=1, noise_std=noise_std)
        self.lb = torch.tensor([-5.0, 0.0], **tkwargs)
        self.ub = torch.tensor([10.0, 15.0], **tkwargs)

        self.a = params.get("a", 1.0)
        self.b = params.get("b", 5.1 / (4.0 * torch.pi ** 2.0))
        self.c = params.get("c", 5.0 / torch.pi)
        self.r = params.get("r", 6.0)
        self.s = params.get("s", 10.0)
        self.t = params.get("t", 1 / (8.0 * torch.pi))

    def _eval(self, x):
        x = torch.atleast_2d(x).to(**tkwargs)
        return (
            self.a * (x[:, 1:2] - self.b * x[:, 0:1] ** 2.0 + self.c * x[:, 0:1] - self.r) ** 2.0
            + self.s * (1 - self.t) * torch.cos(x[:, 0:1])
            + self.s
        )



class BraninHoo(OptimizationProblem):
    def __init__(self, noise_std=0.0, **params):
        super().__init__(ndim=2, nobj=1, noise_std=noise_std)
        self.lb = torch.zeros(self.ndim, **tkwargs)
        self.ub = torch.ones(self.ndim, **tkwargs)

        self.a = params.get("a", 1.0 / 51.95)
        self.b = params.get("b", 5.1 / (4.0 * torch.pi ** 2.0))
        self.c = params.get("c", 5.0 / torch.pi)
        self.r = params.get("r", 6.0)
        self.s = params.get("s", 10.0)
        self.t = params.get("t", 1 / (8.0 * torch.pi))

    def _eval(self, x):
        x = torch.atleast_2d(x).to(**tkwargs)
        return (
            self.a * (15.0 * x[:, 1:2] - self.b * (15.0 * x[:, 0:1] - 5.0) ** 2.0 + self.c * (15.0 * x[:, 0:1] - 5.0) - self.r) ** 2.0
            + self.s * (1 - self.t) * torch.cos((15.0 * x[:, 0:1] - 5.0))
            -44.81
        )