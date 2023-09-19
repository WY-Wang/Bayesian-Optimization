import torch

from BayesianOpt.problem import Rastrigin
from BayesianOpt.model import GaussianProcess
from BayesianOpt.design import QuasiMCDesign
from BayesianOpt.acquisition import EI
from BayesianOpt.base import SurrogateOptimization

RANDOM_SEED = 47
torch.manual_seed(RANDOM_SEED)

prob = Rastrigin(ndim=1)

model = GaussianProcess(
    ndim=prob.ndim,
    lb=prob.lb,
    ub=prob.ub,
    interpolant=False,
)

design = QuasiMCDesign(
    ndim=prob.ndim,
    lb=prob.lb,
    ub=prob.ub,
    random_state=RANDOM_SEED,
)

acquisition = EI(
    ndim=prob.ndim,
    lb=prob.lb,
    ub=prob.ub,
)

algorithm = SurrogateOptimization(
    prob=prob,
    model=model,
    design=design,
    acquisition=acquisition,
)

algorithm.run(
    T=50,
    npts=1,
    maxiter=15000,
    n_restart=10,
    plot_progress=True,
    plot_surrogate=False,
    print_progress=True,
)