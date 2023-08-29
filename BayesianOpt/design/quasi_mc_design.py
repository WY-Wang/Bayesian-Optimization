from torch.quasirandom import SobolEngine


from ..base import ExperimentalDesgin
from ..utils import tkwargs, from_unit_box

class QuasiMCDesign(ExperimentalDesgin):
    def __init__(self, ndim, lb, ub, random_state):
        super().__init__(ndim, lb, ub, random_state)

    def generate_points(self, npts):
        engine = SobolEngine(self.ndim, scramble=True, seed=self.random_state)
        samples = engine.draw(npts).to(**tkwargs)
        return from_unit_box(samples, self.lb, self.ub)