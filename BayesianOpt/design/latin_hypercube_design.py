import pyDOE2
import torch

from ..base import ExperimentalDesgin
from ..utils import tkwargs, from_unit_box

class LatinHypercubeDesign(ExperimentalDesgin):
    def __init__(self, ndim, lb, ub, random_state):
        super().__init__(ndim, lb, ub, random_state)

    def generate_points(self, npts, iterations = 10):
        points = torch.tensor(pyDOE2.lhs(
            n=self.ndim,
            samples=npts,
            iterations=iterations,
            random_state=self.random_state,
        ), **tkwargs)
        return from_unit_box(points, self.lb, self.ub)