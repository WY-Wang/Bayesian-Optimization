import torch

from ..base import ExperimentalDesgin
from ..utils import tkwargs, from_unit_box

class RandomizedDesign(ExperimentalDesgin):
    def __init__(self, ndim, lb, ub, random_state):
        super().__init__(ndim, lb, ub, random_state)

    def generate_points(self, npts):
        return from_unit_box(torch.rand(npts, self.ndim, **tkwargs), self.lb, self.ub)