import torch

from BayesianOpt.problem import Rastrigin
from BayesianOpt.model import ExactGPModel
from BayesianOpt.design import QuasiMCDesign
from BayesianOpt.acquisition import EI
from BayesianOpt.base import SurrogateOptimization

RANDOM_SEED = 47
torch.manual_seed(RANDOM_SEED)

prob = Rastrigin(ndim=1, noise_std=0.1)

model = ExactGPModel(
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
    T=10,
    plot_progress=False,
    print_progress=True,
)