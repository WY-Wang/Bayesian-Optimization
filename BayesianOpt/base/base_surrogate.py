from abc import ABC, abstractmethod
import torch

from BayesianOpt.utils import tkwargs

class Surrogate(ABC):
    def __init__(self, ndim, lb, ub, **options):
        self.ndim = ndim
        self.lb = lb
        self.ub = ub
        self.reset()

    def reset(self):
        self.num_pts = 0
        self.X = torch.empty([0, self.ndim], **tkwargs)
        self.fX = torch.empty([0, 1], **tkwargs)
        self.updated = False

    def add_points(self, x, fx):
        x = torch.atleast_2d(x).to(**tkwargs)
        if isinstance(fx, float):
            fx = torch.tensor([fx], **tkwargs)
        if fx.ndim == 0:
            fx = torch.unsqueeze(fx, dim=0)
        if fx.ndim == 1:
            fx = torch.unsqueeze(fx, dim=1)
        assert x.shape[0] == fx.shape[0] and x.shape[1] == self.ndim

        self.X = torch.vstack((self.X, x))
        self.fX = torch.vstack((self.fX, fx))
        self.num_pts += x.shape[0]
        self.updated = False

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError