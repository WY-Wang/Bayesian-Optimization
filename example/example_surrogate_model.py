import torch
import matplotlib.pyplot as plt

from BayesianOpt.model import GaussianProcess
from BayesianOpt.utils import tkwargs, from_unit_box
from BayesianOpt.problem import Simple

RANDOM_SEED = 47
torch.manual_seed(RANDOM_SEED)

prob = Simple(
    ndim=1,
    noise_std=0.0,
)

model = GaussianProcess(
    ndim=prob.ndim,
    lb=prob.lb,
    ub=prob.ub,
    interpolant=True,
)

test_x = torch.unsqueeze(torch.linspace(prob.lb[0], prob.ub[0], 10000), dim=1).to(**tkwargs)
test_y = prob._eval(test_x)

train_x = torch.unsqueeze(from_unit_box(torch.rand(6), prob.lb, prob.ub), dim=1).to(**tkwargs)
train_y = prob.eval(train_x)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

model.add_points(train_x, train_y)
pred = model.predict(test_x)

ax.plot(torch.squeeze(test_x).numpy(), torch.squeeze(test_y).numpy(), "b-", label="True")
ax.plot(torch.squeeze(test_x).numpy(), torch.squeeze(pred["mean"]).numpy(), "r-", label=model.__class__.__name__)

lower, upper = pred["dist"].confidence_region()
ax.fill_between(torch.squeeze(test_x).numpy(), torch.squeeze(lower).numpy(), torch.squeeze(upper).numpy(), alpha=0.5)
for _ in range(5):
    ax.plot(torch.squeeze(test_x).numpy(), torch.squeeze(pred["dist"].sample(torch.Size([1]))).numpy(), "y--")

ax.plot(torch.squeeze(train_x).numpy(), torch.squeeze(train_y).numpy(), "k*", label="Observations")

ax.set_xlabel("x")
ax.set_ylabel("f(x)")

ax.legend()
plt.show()