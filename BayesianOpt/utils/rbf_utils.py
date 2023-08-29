import numpy as np



class CubicKernel:
    def __init__(self):
        self.order = 2

    def eval(self, dists):
        return dists ** 3

    def deriv(self, dists):
        return 3 * dists ** 2



class LinearKernel:
    def __init__(self):
        self.order = 1

    def eval(self, dists):
        return dists

    def deriv(self, dists):
        return np.ones(dists.shape)



class TPSKernel:
    def __init__(self):
        self.order = 2

    def eval(self, dists):
        dists[dists < np.finfo(float).eps] = np.finfo(float).eps
        return (dists ** 2) * np.log(dists)

    def deriv(self, dists):
        dists[dists < np.finfo(float).eps] = np.finfo(float).eps
        return dists * (1 + 2 * np.log(dists))


class ConstantTail:
    def __init__(self, ndim):
        self.degree = 0
        self.ndim = ndim
        self.ndim_tail = 1

    def eval(self, x):
        x = np.atleast_2d(x)
        return np.ones((x.shape[0], 1))

    def deriv(self, x):
        x = np.atleast_2d(x)
        return np.zeros((x.shape[1], 1))



class LinearTail:
    def __init__(self, ndim):
        self.degree = 1
        self.ndim = ndim
        self.ndim_tail = 1 + ndim

    def eval(self, x):
        x = np.atleast_2d(x)
        return np.hstack((np.ones((x.shape[0], 1)), x))

    def deriv(self, x):
        x = np.atleast_2d(x)
        return np.hstack((np.zeros((x.shape[1], 1)), np.eye((x.shape[1]))))
