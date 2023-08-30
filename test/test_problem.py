import torch

from BayesianOpt.utils import tkwargs
from BayesianOpt.problem import *


prob = Ackley(ndim=6)
x = torch.zeros(prob.ndim, **tkwargs)
fx = torch.tensor([[0.0]], **tkwargs)
assert torch.allclose(prob.eval(x), fx, atol=1e-5), f"Check {prob.name}"


prob = Branin()
x = torch.tensor([
    [-torch.pi, 12.275],
    [torch.pi, 2.275],
    [9.42478, 2.475],
], **tkwargs)
fx = torch.tensor([[0.397887], [0.397887], [0.397887]], **tkwargs)
assert torch.allclose(prob.eval(x), fx, atol=1e-5), f"Check {prob.name}"


prob = Rastrigin(ndim=6)
x = torch.zeros(prob.ndim, **tkwargs)
fx = torch.tensor([[0.0]], **tkwargs)
assert torch.allclose(prob.eval(x), fx, atol=1e-5), f"Check {prob.name}"


prob = Hartmann3()
x = torch.tensor([0.114614, 0.555649, 0.852547], **tkwargs)
fx = torch.tensor([[-3.86278]], **tkwargs)
assert torch.allclose(prob.eval(x), fx, atol=1e-5), f"Check {prob.name}"


prob = Hartmann6()
x = torch.tensor([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573], **tkwargs)
fx = torch.tensor([[-3.32237]], **tkwargs)
assert torch.allclose(prob.eval(x), fx, atol=1e-5), f"Check {prob.name}"