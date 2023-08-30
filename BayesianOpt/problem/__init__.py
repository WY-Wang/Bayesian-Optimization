from .simple import Simple
from .ackley import Ackley
from .rastrigin import Rastrigin
from .branin import Branin, BraninHoo
from .hartmann import Hartmann3, Hartmann4, Hartmann6
from .dtlz import DTLZ
from .zdt import ZDT

__all__ = [
    "Simple",
    "Ackley",
    "Rastrigin",
    "Branin",
    "BraninHoo",
    "Hartmann3",
    "Hartmann4",
    "Hartmann6",

    "DTLZ",
    "ZDT",
]