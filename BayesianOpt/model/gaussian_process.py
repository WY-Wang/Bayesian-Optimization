import torch
import gpytorch
from gpytorch.models import ExactGP
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.constraints import Interval, GreaterThan

from ..base import Surrogate
from ..utils import tkwargs, to_unit_box



class GPCoreModel(ExactGP, GPyTorchModel):
    def __init__(
        self,
        train_x,
        train_y,
        mean_module,
        covar_module,
        likelihood,
        interpolant,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

        if interpolant:
            self.likelihood.noise = 1e-4
            self.likelihood.raw_noise.requires_grad = False

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



class GaussianProcess(Surrogate):
    def __init__(
        self,
        ndim: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        **options,
    ):
        super().__init__(ndim=ndim, lb=lb, ub=ub)
        self.mean_module = options.get("mean_module",
            gpytorch.means.ConstantMean())
        self.covar_module = options.get("covar_module",
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(
                    nu=2.5, ard_num_dims=self.ndim, lengthscale_constraint=Interval(0.005, 4.0)
                )
            ))
        self.likelihood = options.get("likelihood",
            gpytorch.likelihoods.GaussianLikelihood(noise_constraint=Interval(1e-6, 1e-2)))

        self.optimizer = options.get("optimizer")
        self.mll = options.get("mll")
        self.epochs = options.get("epochs", 100)
        self.interpolant = options.get("interpolant", False)

    def reset(self):
        super().reset()

    def fit(self, optimize):
        if not self.updated:
            self._X = to_unit_box(self.X, self.lb, self.ub)
            self.model = GPCoreModel(
                train_x=self._X,
                train_y=self.fX[:, 0],
                mean_module=self.mean_module,
                covar_module=self.covar_module,
                likelihood=self.likelihood,
                interpolant=self.interpolant,
            ).to(**tkwargs)
            self.model.train()

            if optimize:
                if self.optimizer is None:
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
                if self.mll is None:
                    self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
    
                max_cholesky_size = float("inf")
                with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                    for i in range(self.epochs):
                        self.optimizer.zero_grad()
    
                        output = self.model(self._X)
                        loss = - self.mll(output, self.fX[:, 0])
                        loss.backward()
    
                        # print(f"Epoch {i + 1}/{self.epochs} - \tloss: {loss.item():.3f}, "
                        #       f"\tlengthscale: {self.model.covar_module.base_kernel.lengthscale.item():.3f} "
                        #       f"\tnoise: {self.model.likelihood.noise.item():.3f}"
                        # )
    
                        self.optimizer.step()
    
                self.updated = True

    def predict(self, x, optimize=True):
        self.fit(optimize=optimize)

        x = to_unit_box(torch.atleast_2d(x).to(**tkwargs), self.lb, self.ub)
        self.model.eval()

        pred = {}
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_y = self.model.posterior(x, observation_noise=False).mvn
            pred["dist"] = pred_y
            pred["mean"] = torch.unsqueeze(pred_y.mean, dim=1)
            pred["variance"] = torch.unsqueeze(pred_y.variance, dim=1)
            return pred