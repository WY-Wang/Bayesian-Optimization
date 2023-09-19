import torch

def to_unit_box(x, lb = None, ub = None):
    if lb is None:
        lb = torch.amin(x, dim=0, keepdim=False)
    if ub is None:
        ub = torch.amax(x, dim=0, keepdim=False)
    return (x - lb) / (ub - lb)

def from_unit_box(x, lb, ub):
    return (ub - lb) * x + lb