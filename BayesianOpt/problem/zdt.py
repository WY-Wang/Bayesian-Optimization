import torch
from pymoo.factory import get_problem

from ..utils import tkwargs
from ..base import OptimizationProblem

class ZDT(OptimizationProblem):
    def __init__(self, ndim, ordinal):
        self.super_prob = get_problem("ZDT" + str(ordinal), n_var=ndim)
        super().__init__(ndim=self.super_prob.n_var, nobj=self.super_prob.n_obj)
        self.ordinal = ordinal
        self.name += str(ordinal)

        self.lb = torch.tensor(self.super_prob.xl, **tkwargs)
        self.ub = torch.tensor(self.super_prob.xu, **tkwargs)

    def eval(self, x):
        x = torch.atleast_2d(x).to(**tkwargs)
        out = {"F": None, "CV": None, "G": None, "dF": None, "dG": None}
        self.super_prob._evaluate(x.numpy(), out=out, return_as_dictionary=True)
        return torch.tensor(out["F"], **tkwargs)

    def pareto_front(self):
        return self.super_prob.pareto_front()