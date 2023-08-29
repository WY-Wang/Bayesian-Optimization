import torch
import numpy as np
from scipy.stats import norm

from ..utils import tkwargs
from ..base import Acquisition



class EI(Acquisition):
    def __init__(self, ndim, lb, ub, dtol=1e-3):
        super().__init__(ndim=ndim, lb=lb, ub=ub)
        self.dtol = dtol

    def merit(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, **tkwargs)
        x = torch.atleast_2d(x)

        pred = self.model.predict(x)
        gamma = (torch.amin(self.model.fX) - pred["mean"]) / (pred["variance"] ** 0.5)
        ei = pred["variance"] ** 0.5 * (gamma * norm.cdf(gamma) + norm.pdf(gamma))

        if self.dtol > 0.0:
            xx = torch.vstack((self.model.X, self.points))
            dists = torch.cdist(x, xx)
            dmerit = torch.amin(dists, dim=1, keepdim=True)
            ei[dmerit < self.dtol] = torch.tensor(-np.inf, **tkwargs)

        return -ei