import torch
import numpy as np

from ..utils import tkwargs
from ..base import Acquisition



class UCB(Acquisition): # or LCB
    def __init__(self, ndim, lb, ub, beta=2.0, dtol=1e-3, minimize=True):
        super().__init__(ndim=ndim, lb=lb, ub=ub)
        self.beta = beta
        self.dtol = dtol
        self.minimize = minimize
        self.reverse = False

    def merit(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, **tkwargs)
        x = torch.atleast_2d(x)

        pred = self.model.predict(x)
        cb = pred["mean"] - (self.beta ** 0.5) * (pred["variance"] ** 0.5)

        if self.dtol > 0.0:
            xx = torch.vstack((self.model.X, self.points))
            dists = torch.cdist(x, xx)
            dmerit = torch.amin(dists, dim=1, keepdim=True)
            cb[dmerit < self.dtol] = torch.tensor(np.inf, **tkwargs)

        return cb.cpu()