import torch
import numpy as np
import scipy.spatial as scpspatial
import scipy.linalg as scplinalg

from ..base import Surrogate
from ..utils import LinearKernel, CubicKernel, TPSKernel
from ..utils import ConstantTail, LinearTail
from ..utils import tkwargs, to_unit_box



class ExactRBFModel(Surrogate):
    # This RBF implementation follows the RBFInterpolant class from the pySOT package "David Eriksson,
    # David Bindel, Christine A. Shoemaker. pySOT and POAP: An event-driven asynchronous framework for
    # surrogate optimization. arXiv preprint arXiv:1908.00420, 2019"

    def __init__(self, ndim, lb, ub, **options):
        super().__init__(ndim=ndim, lb=lb, ub=ub)

        self.eta = options.get("eta", 1e-6)
        self.kernel = {
            "Linear": LinearKernel(),
            "Cubic": CubicKernel(),
            "TPS": TPSKernel(),
        }[options.get("kernel", "Cubic")]
        self.tail = {
            "Constant": ConstantTail(ndim=ndim),
            "Linear": LinearTail(ndim=ndim),
        }[options.get("tail", "Linear")]

    def reset(self):
        super().reset()

        self.L = None
        self.U = None
        self.piv = None
        self.c = None

    def fit(self):
        if not self.updated:
            n = self.num_pts
            ntail = self.tail.ndim_tail
            nact = ntail + n
            self._X = to_unit_box(self.X, self.lb, self.ub).numpy()

            if self.c is None:  # Initial fit
                assert self.num_pts >= ntail

                X = self._X[0:n, :]
                D = scpspatial.distance.cdist(X, X)
                Phi = self.kernel.eval(D) + self.eta * np.eye(n)
                P = self.tail.eval(X)

                # Set up the systems matrix
                A1 = np.hstack((np.zeros((ntail, ntail)), P.T))
                A2 = np.hstack((P, Phi))
                A = np.vstack((A1, A2))

                [LU, piv] = scplinalg.lu_factor(A)
                self.L = np.tril(LU, -1) + np.eye(nact)
                self.U = np.triu(LU)

                # Construct the usual pivoting vector so that we can increment
                self.piv = np.arange(0, nact)
                for i in range(nact):
                    self.piv[i], self.piv[piv[i]] = self.piv[piv[i]], self.piv[i]

            else:  # Extend LU factorization
                k = self.c.shape[0] - ntail
                numnew = n - k
                kact = ntail + k

                X = self._X[:n, :]
                XX = self._X[k:n, :]
                D = scpspatial.distance.cdist(X, XX)
                Pnew = np.vstack((self.tail.eval(XX).T, self.kernel.eval(D[:k, :])))
                Phinew = self.kernel.eval(D[k:, :]) + self.eta * np.eye(numnew)

                L21 = np.zeros((kact, numnew))
                U12 = np.zeros((kact, numnew))
                for i in range(numnew):  # TODO: Can we use level-3 BLAS?
                    L21[:, i] = scplinalg.solve_triangular(a=self.U, b=Pnew[:kact, i], lower=False, trans="T")
                    U12[:, i] = scplinalg.solve_triangular(a=self.L, b=Pnew[self.piv[:kact], i], lower=True, trans="N")
                L21 = L21.T
                try:  # Compute Cholesky factorization of the Schur complement
                    C = scplinalg.cholesky(a=Phinew - np.dot(L21, U12), lower=True)
                except:  # Compute a new LU factorization if Cholesky fails
                    self.c = None
                    return self.fit()

                self.piv = np.hstack((self.piv, np.arange(kact, nact)))
                self.L = np.vstack((self.L, L21))
                L2 = np.vstack((np.zeros((kact, numnew)), C))
                self.L = np.hstack((self.L, L2))
                self.U = np.hstack((self.U, U12))
                U2 = np.hstack((np.zeros((numnew, kact)), C.T))
                self.U = np.vstack((self.U, U2))

            # Update coefficients
            fX = self.fX.numpy().copy()
            rhs = np.vstack((np.zeros((ntail, 1)), fX))
            self.c = scplinalg.solve_triangular(a=self.L, b=rhs[self.piv], lower=True)
            self.c = scplinalg.solve_triangular(a=self.U, b=self.c, lower=False)
            self.updated = True

    def predict(self, x):
        pred = {}
        self.fit()

        _x = to_unit_box(torch.atleast_2d(x), self.lb, self.ub)
        ds = scpspatial.distance.cdist(_x, self._X)
        ntail = self.tail.ndim_tail

        pred["mean"] = torch.tensor(np.dot(self.kernel.eval(ds), self.c[ntail: ntail + self.num_pts]) +
                                    np.dot(self.tail.eval(_x), self.c[:ntail]), **tkwargs)
        return pred
