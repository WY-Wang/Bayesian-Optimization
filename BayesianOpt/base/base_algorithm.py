from abc import ABC, abstractmethod
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt


class SurrogateOptimization(ABC):
    def __init__(self, prob, model, design, acquisition, ninits=None):
        self.prob = prob
        self.model = model
        self.design = design
        self.acquisition = acquisition
        self.name = self.__class__.__name__

        self.ninits = ninits if ninits else 2 * (prob.ndim + 1)

        self.X = torch.empty(size=(0, self.prob.ndim))
        self.fX = torch.empty(size=(0, self.prob.nobj))

        self.xbest = None
        self.fxbest = None

        self.time = time.time()
        self.nevals = 0
        self.initialized = False

    def reset(self):
        self.X = torch.empty(size=(0, self.prob.ndim))
        self.fX = torch.empty(size=(0, self.prob.nobj))

        self.xbest = None
        self.fxbest = None

        self.time = time.time()
        self.nevals = 0
        self.initialized = False

    def initialization(self):
        self.initialized = True

        x = self.design.generate_points(npts=self.ninits)
        fx = self.prob.eval(x)

        self.model.add_points(x=x, fx=fx)
        self.X = torch.vstack((self.X, x))
        self.fX = torch.vstack((self.fX, fx))

        index = torch.squeeze(torch.argmin(self.fX, dim=0, keepdim=True))
        self.xbest = self.X[index, :]
        self.fxbest = self.fX[index, :]
        self.nevals = self.ninits

    def run(self, T, npts=1, n_restart=10, plot_surrogate=False, plot_progress=False, print_progress=False):
        if not self.initialized:
            self.initialization()

        while self.nevals < T:
            self._run(
                npts=npts,
                n_restart=n_restart,
                plot_surrogate=plot_surrogate,
                plot_progress=plot_progress,
                print_progress=print_progress,
            )

    def _run(self, npts, n_restart, plot_surrogate, plot_progress, print_progress):
        x = self.acquisition.optimize(
            model=self.model,
            npts=npts,
            n_restarts=n_restart
        )
        fx = self.prob.eval(x)
        self.nevals += x.shape[0]

        self.model.add_points(x=x, fx=fx)
        if plot_surrogate: self.model.plot(bound=True)
        self.X = torch.vstack((self.X, x))
        self.fX = torch.vstack((self.fX, fx))

        _x = torch.vstack((x, torch.atleast_2d(self.xbest)))
        _fx = torch.vstack((fx, torch.atleast_2d(self.fxbest)))
        index = torch.squeeze(torch.argmin(_fx, dim=0, keepdim=True))
        self.xbest = _x[index, :]
        self.fxbest = _fx[index, :]

        if print_progress:
            print(f"Evaluations = {self.nevals}: \n"
                  f"\tfx = {fx.numpy()}, x = {x.numpy()} \n"
                  f"\tfxbest = {self.fxbest.numpy()}, xbest = {self.xbest.numpy()}")

        if plot_progress: self.plot()

        return x, fx

    def plot(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        if self.prob.ndim == 1:
            self.model.plot(ax=ax, bound=True)
        elif self.prob.ndim == 2:
            nrows, ncols = 200, 200
            test_x = np.linspace(self.prob.lb[0], self.prob.ub[0], nrows)
            test_y = np.linspace(self.prob.lb[1], self.prob.ub[1], ncols)
            test_xy = torch.tensor([[_x, _y] for _y in test_y for _x in test_x])
            test_z = self.prob.eval(test_xy).reshape((ncols, nrows))

            ct = ax.contour(test_x, test_y, test_z, 50, cmap="binary_r")
            fig.colorbar(ct, ax=ax)

            ax.plot(torch.squeeze(self.X[:, 0]).numpy(), torch.squeeze(self.X[:, 1]).numpy(), "k*", label="Observations")
            ax.plot(torch.squeeze(self.X[-1, 0]).numpy(), torch.squeeze(self.X[-1, 1]).numpy(), "b*", label="New")

            ax.set_xlabel("x1")
            ax.set_ylabel("x2")

        plt.show()

    def save_results(self, root, trial_id):
        result = torch.hstack((self.X, self.fX)).numpy()

        path = root
        for dirname in [self.name, self.prob.name, self.prob.ndim]:
            path = path + f"/{dirname}"
            if not os.path.exists(path):
                os.mkdir(path)

        filename = f"{self.name}_{self.prob.name}_{self.prob.ndim}_{trial_id}.txt"
        np.savetxt(f"{path}/{filename}", result)