from abc import ABC, abstractmethod

import torch
import scipy.optimize as scpopt

from ..utils import tkwargs, from_unit_box

class Acquisition(ABC):
    def __init__(self, ndim, lb, ub):
        self.ndim = ndim
        self.lb = lb
        self.ub = ub
        self.points = torch.empty((0, self.ndim), **tkwargs)

    def optimize(self, model, npts, maxiter, n_restarts):
        self.model = model
        self.points = torch.empty((0, self.ndim), **tkwargs)

        for _ in range(npts):
            x = None
            for _ in range(n_restarts + 1):
                result = scpopt.minimize(
                    fun=self.merit,
                    x0=from_unit_box(torch.rand(self.ndim, **tkwargs), self.lb, self.ub),
                    method="L-BFGS-B",
                    bounds=torch.vstack((self.lb, self.ub)).T,
                    options={"maxiter": maxiter},
                )
                _x = result.x.copy()
                if x is None or self.merit(x) > self.merit(_x): x = _x
            x = torch.tensor(x).to(**tkwargs)
            self.points = torch.vstack((self.points, x))

        return self.points

    def optimize_de(self, model, npts, popsize, maxgen):
        self.model = model
        self.points = torch.empty((0, self.ndim), **tkwargs)

        def func(x):
            return self.merit(x).numpy()

        for _ in range(npts):
            result = scpopt.differential_evolution(
                func=func,
                bounds=torch.vstack((self.lb, self.ub)).T.numpy(),
                popsize=popsize,
                maxiter=maxgen,
                seed=None,
            )

            x = torch.tensor(result.x.copy()).to(**tkwargs)
            self.points = torch.vstack((self.points, x))

        return self.points

    @abstractmethod
    def merit(self, x):
        raise NotImplementedError