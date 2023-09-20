import torch
import matplotlib.pyplot as plt

from BayesianOpt.utils import tkwargs
from BayesianOpt.design import QuasiMCDesign

RANDOM_SEED = 47
torch.manual_seed(RANDOM_SEED)

N_DIM = 2
LOWER = torch.zeros(N_DIM, **tkwargs)
UPPER = torch.ones(N_DIM, **tkwargs)

design = QuasiMCDesign(
    ndim=N_DIM,
    lb=LOWER,
    ub=UPPER,
    random_state=RANDOM_SEED,
)

fig, axes = plt.subplots(1, 1, figsize=(8, 6))

samples = design.generate_points(npts=10)
axes.plot(samples[:, 0].numpy(), samples[:, 1].numpy(), "ko")

samples = design.generate_points(npts=20)
axes.plot(samples[:, 0].numpy(), samples[:, 1].numpy(), "b*")

plt.show()