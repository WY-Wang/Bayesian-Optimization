import torch

from ..utils import tkwargs
from ..base import OptimizationProblem


class Hartmann3(OptimizationProblem):
    def __init__(self, noise_std=0.0):
        super().__init__(ndim=3, nobj=1, noise_std=noise_std)
        self.lb = torch.zeros(self.ndim, **tkwargs)
        self.ub = torch.ones(self.ndim, **tkwargs)

        self.a = torch.tensor([1.0, 1.2, 3.0, 3.2], **tkwargs)
        self.A = torch.tensor([
            [3.0, 10, 30],
            [0.1, 10, 35],
            [3.0, 10, 30],
            [0.1, 10, 35]
        ], **tkwargs)
        self.B = torch.tensor([
            [3689.0, 1170.0, 2673.0],
            [4699.0, 4387.0, 7470.0],
            [1091.0, 8732.0, 5547.0],
            [381.0, 5743.0, 8828.0],
        ], **tkwargs).mul(1e-4)

    def _eval(self, x):
        x = torch.atleast_2d(x).to(**tkwargs)
        inner_sum = torch.sum(self.A * (x.unsqueeze(-2) - self.B) ** 2, dim=-1)
        return -torch.sum(self.a * torch.exp(-inner_sum), dim=-1, keepdim=True)


class Hartmann4(OptimizationProblem):
    def __init__(self, noise_std=0.0):
        super().__init__(ndim=4, nobj=1, noise_std=noise_std)
        self.lb = torch.zeros(self.ndim, **tkwargs)
        self.ub = torch.ones(self.ndim, **tkwargs)

        self.a = torch.tensor([1.0, 1.2, 3.0, 3.2], **tkwargs)
        self.A = torch.tensor([
            [10.0, 3.0, 17.0, 3.5],
            [0.05, 10.0, 17.0, 0.1],
            [3.0, 3.5, 1.7, 10.0],
            [17.0, 8.0, 0.05, 10.0],
        ], **tkwargs)
        self.B = torch.tensor([
            [1312.0, 1696.0, 5569.0, 124.0],
            [2329.0, 4135.0, 8307.0, 3736.0],
            [2348.0, 1451.0, 3522.0, 2883.0],
            [4047.0, 8828.0, 8732.0, 5743.0],
        ], **tkwargs).mul(1e-4)

    def _eval(self, x):
        x = torch.atleast_2d(x).to(**tkwargs)
        inner_sum = torch.sum(self.A * (x.unsqueeze(-2) - self.B) ** 2, dim=-1)
        return (1.1 - torch.sum(self.a * torch.exp(-inner_sum), dim=-1, keepdim=True)) / 0.839


class Hartmann6(OptimizationProblem):
    def __init__(self, noise_std=0.0):
        super().__init__(ndim=6, nobj=1, noise_std=noise_std)
        self.lb = torch.zeros(self.ndim, **tkwargs)
        self.ub = torch.ones(self.ndim, **tkwargs)

        self.a = torch.tensor([1.0, 1.2, 3.0, 3.2], **tkwargs)
        self.A = torch.tensor([
            [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
        ], **tkwargs)
        self.B = torch.tensor([
            [1312.0, 1696.0, 5569.0, 124.0, 8283.0, 5886.0],
            [2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0],
            [2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0],
            [4047.0, 8828.0, 8732.0, 5743.0, 1091.0, 381.0],
        ], **tkwargs).mul(1e-4)

    def _eval(self, x):
        x = torch.atleast_2d(x).to(**tkwargs)
        inner_sum = torch.sum(self.A * (x.unsqueeze(-2) - self.B) ** 2, dim=-1)
        return -torch.sum(self.a * torch.exp(-inner_sum), dim=-1, keepdim=True)