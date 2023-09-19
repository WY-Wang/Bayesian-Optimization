import torch
import numpy as np
import matplotlib.pyplot as plt

from BayesianOpt.problem import Branin

RANDOM_SEED = 47
torch.manual_seed(RANDOM_SEED)

problem = Branin()

nrows, ncols = 200, 200
x = np.linspace(problem.lb[0], problem.ub[0], nrows)
y = np.linspace(problem.lb[1], problem.ub[1], ncols)
xy = torch.tensor([[_x, _y] for _y in y for _x in x])
z = problem.eval(xy).reshape((ncols, nrows))

fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)

ct = ax.contour(x, y, z, 50, cmap="binary_r")
fig.colorbar(ct, ax=ax)

ax.set_xlabel("x1")
ax.set_ylabel("x2")
plt.show()