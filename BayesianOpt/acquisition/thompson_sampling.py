import torch
import numpy as np

from ..utils import tkwargs
from ..base import Acquisition
from ..design import LatinHypercubeDesign



class TS(Acquisition):
    def __init__(self, ndim, lb, ub, dtol=1e-3):
        super().__init__(ndim=ndim, lb=lb, ub=ub)
        self.dtol = dtol

    def optimize(self, model, npts, maxiter, n_restarts, **options):
        self.model = model
        self.design = LatinHypercubeDesign(ndim=self.ndim, lb=self.lb, ub=self.ub, random_state=None)

        xx = self.design.generate_points(npts=maxiter)
        ts = self.merit(xx)

        self.points = torch.empty((0, self.ndim), **tkwargs)
        for _ in range(npts):
            index = torch.argmin(ts)
            x = xx[index, :]
            dists = torch.cdist(xx, torch.atleast_2d(x))
            dmerit = torch.amin(dists, dim=1, keepdim=True)
            ts[dmerit < self.dtol] = torch.tensor(np.inf, **tkwargs)
            self.points = torch.vstack((self.points, x))
            print(self.points)
        return self.points

    def merit(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, **tkwargs)
        x = torch.atleast_2d(x)

        pred = self.model.predict(x)
        ts = pred["dist"].sample(torch.Size([1])).reshape(x.shape[0], -1)

        if self.dtol > 0.0:
            xx = torch.vstack((self.model.X, self.points))
            dists = torch.cdist(x, xx)
            dmerit = torch.amin(dists, dim=1, keepdim=True)
            ts[dmerit < self.dtol] = torch.tensor(np.inf, **tkwargs)

        return ts