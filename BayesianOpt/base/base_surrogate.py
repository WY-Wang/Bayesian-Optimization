from abc import ABC, abstractmethod

import torch
import matplotlib.pyplot as plt

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

    def plot(self, ax=None, **options):
        test_x = torch.unsqueeze(torch.linspace(self.lb[0], self.ub[0], 1000), dim=1).to(**tkwargs)
        pred = self.predict(test_x)

        _fig, _ax = (None, ax) if ax else plt.subplots(1, 1, figsize=(4, 3))

        _ax.plot(torch.squeeze(self.X).numpy(), torch.squeeze(self.fX).numpy(), "k*", label="Observations")
        _ax.plot(torch.squeeze(test_x).numpy(), torch.squeeze(pred["mean"]).numpy(), "b-", label=self.__class__.__name__)
        if options.get("bound"):
            try:
                _ax.fill_between(torch.squeeze(test_x).numpy(), torch.squeeze(pred["lowerbound"]).numpy(), torch.squeeze(pred["upperbound"]).numpy(), alpha=0.5)
            except:
                print(f"\"bound\" is not defined for {self.__class__.__name__}")
        if options.get("sample"):
            try:
                for _ in range(options.get("sample")):
                    _ax.plot(torch.squeeze(test_x).numpy(), torch.squeeze(pred["dist"].sample(torch.Size([1]))).numpy(), "r-")
            except:
                print(f"\"sample\" is not defined for {self.__class__.__name__}")

        _ax.legend()
        if not ax: plt.show()

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError