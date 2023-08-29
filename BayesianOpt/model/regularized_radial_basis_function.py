import torch
import numpy as np
import scipy.spatial as scpspatial
import scipy.linalg as scplinalg
import scipy.optimize as scpopt
import scipy.special as scpspecial

from ..base import Surrogate
from ..utils import LinearKernel, CubicKernel, TPSKernel
from ..utils import ConstantTail, LinearTail
from ..utils import tkwargs, to_unit_box



class NoisyRBFModel(Surrogate):
    # This implementation follows "Shen, Y., & Shoemaker, C. A. (2020, December). Global optimization for
    # noisy expensive black-box multi-modal functions via radial basis function surrogate. In 2020 Winter
    # Simulation Conference (WSC) (pp. 3020-3031). IEEE."

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

            # Initial fit
            assert self.num_pts >= ntail

            X = self._X[0:n, :]
            D = scpspatial.distance.cdist(X, X)
            Phi = self.kernel.eval(D) + self.eta * np.eye(n)
            P = self.tail.eval(X)

            # Set up the systems matrix
            A1 = np.hstack((np.zeros((ntail, ntail)), P.T))
            A2 = np.hstack((P, Phi))
            A = np.vstack((A1, A2))

            Q1 = np.hstack((np.zeros((ntail, ntail)), np.zeros((ntail, self.X.shape[0]))))
            Q2 = np.hstack((np.zeros((self.X.shape[0], ntail)), Phi / float(self.X.shape[0])))
            Q = np.vstack((Q1, Q2))

            [LU, piv] = scplinalg.lu_factor(A.T @ A + Q)
            self.L = np.tril(LU, -1) + np.eye(nact)
            self.U = np.triu(LU)

            # Construct the usual pivoting vector so that we can increment
            self.piv = np.arange(0, nact)
            for i in range(nact):
                self.piv[i], self.piv[piv[i]] = self.piv[piv[i]], self.piv[i]

            # Update coefficients
            fX = self.fX.numpy().copy()
            rhs = np.vstack((np.zeros((ntail, 1)), fX))
            rhs = A.T @ rhs
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



class HomoRBFModel(Surrogate):
    # This implementation is based on "Ji, Y., & Kim, S. (2014, December). Regularized radial basis function models
    # for stochastic simulation. In Proceedings of the Winter Simulation Conference 2014 (pp. 3833-3844). IEEE.",
    # and,
    # "JI YIBO (2014-01-17). METAMODEL-BASED GLOBAL OPTIMIZATION AND GENERALIZED NASH EQUILIBRIUM PROBLEMS.
    # ScholarBank@NUS Repository."
    #
    # We further consider the standard deviation of noises as part of regularization parameter \mu to make this
    # surrogate model fitting without known noise level.

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

        # strategy = "N": no updating strategy on matrix inversion (small error but high computational cost);
        # strategy = "S": update matrix inversion with a single point each time;
        # strategy = "M": update matrix inversion with multiple points each time.
        self.strategy = options.get("strategy", "N")
        self.bound = options.get("bound", (-6.0, -1.0)) # Will be transformed to 10^bound when tuning regularization parameter
        self.n_restarts = options.get("n_restarts", 0)

    def reset(self):
        super().reset()

        self.L = None
        self.U = None
        self.piv = None
        self.c = None

        self.mu = None
        self.update_times = 0

    def fit(self):
        if not self.updated:
            n = self.num_pts
            ntail = self.tail.ndim_tail
            self._X = to_unit_box(self.X, self.lb, self.ub).numpy()

            X = self._X[0:n, :]
            Phi = self.kernel.eval(scpspatial.distance.cdist(X, X)) + self.eta * np.eye(n)
            Pi = self.tail.eval(X)

            # todo: Compute (or Update) A_inv, i.e., the matrix inversion of A
            if self.strategy == "N" or self.update_times % 50 == 0:
                assert self.num_pts >= ntail
                A1 = np.hstack((Phi, Pi))
                A2 = np.hstack((Pi.T, np.zeros((ntail, ntail))))
                self.A = np.vstack((A1, A2))
                self.A_inv = scplinalg.inv(self.A)
                # print(np.allclose(np.matmul(self.A, self.A_inv), np.eye(self.A.shape[0])))
                # print(np.amax(np.abs(np.matmul(self.A, self.A_inv) - np.eye(self.A.shape[0]))))
                # print(np.allclose(np.matmul(self.A, scplinalg.inv(self.A)), np.eye(self.A.shape[0])))
                # print(np.amax(np.abs(np.matmul(self.A, scplinalg.inv(self.A)) - np.eye(self.A.shape[0]))))
            else:
                k = self.c.shape[0] - ntail
                numnew = n - k if self.strategy == "M" else 1
                while k < n:
                    A1 = np.hstack((Phi[:k + numnew, :k + numnew], Pi[:k + numnew, :]))
                    A2 = np.hstack((Pi[:k + numnew, :].T, np.zeros((ntail, ntail))))
                    self.A = np.vstack((A1, A2))

                    # Construct perturbation matrix
                    _P = np.zeros((numnew + ntail, numnew + ntail))
                    for i in range(ntail): _P[i, i + numnew] = 1
                    for i in range(ntail, numnew + ntail): _P[i, i - ntail] = 1

                    P = np.vstack((
                        np.hstack((np.eye(k), np.zeros((k, numnew + ntail)))),
                        np.hstack((np.zeros((numnew + ntail, k)), _P))
                    ))

                    _A = np.matmul(np.matmul(P, self.A), P.T)
                    B = _A[:k + ntail, k + ntail:]
                    C = _A[k + ntail:, :k + ntail]
                    D = _A[k + ntail:, k + ntail:]

                    # Block matriix inversion
                    CA_inv = np.matmul(C, self.A_inv)
                    A_invB = np.matmul(self.A_inv, B)
                    U = scplinalg.inv(np.subtract(D, np.matmul(CA_inv, B)))

                    _A_inv = np.vstack((
                        np.hstack((
                            np.add(self.A_inv, np.matmul(np.matmul(A_invB, U), CA_inv)),
                            -np.matmul(A_invB, U)
                        )),
                        np.hstack((-np.matmul(U, CA_inv), U)),
                    ))
                    self.A_inv = np.matmul(np.matmul(P.T, _A_inv), P)
                    k += numnew

            # todo: Regularization parameter estimation

            # todo: 1) Consider the unknown noise level as part of the regularization parameter;
            #       3) Ensure that self.A_inv @ Sigma has real eigen values and vectors by adding a small eta;

            Sigma = np.vstack((
                np.hstack((np.eye(n), np.zeros((n, ntail)))),
                np.hstack((np.zeros((ntail, n)), np.eye(ntail) * self.eta))
            ))

            delta, G = scplinalg.eig(np.matmul(self.A_inv, Sigma))
            delta, G, G_inv = np.real(delta), np.real(G), np.real(scplinalg.inv(G))

            fX = self.fX.numpy().copy()
            rhs = np.vstack((fX, np.zeros((ntail, 1))))

            pre_alpha = np.matmul(G_inv, self.A_inv)
            def LOOCV(mu):
                mu = np.atleast_1d(np.power(10, mu))
                _r_A_inv = np.matmul(np.matmul(G, np.diag(1.0 / (1.0 + n * mu[0] * delta))), pre_alpha)
                _c = np.matmul(_r_A_inv, rhs)
                f = sum([np.abs(_c[i, 0]) / np.abs(_r_A_inv[i, i]) for i in range(n)]) / n
                return f

            self.mu = np.random.uniform(self.bound[0], self.bound[1]) if self.mu is None else np.log10(self.mu)
            self.mu = scpopt.minimize(LOOCV, np.array((self.mu,)), method="L-BFGS-B", bounds=(self.bound,)).x[0]
            target = LOOCV(self.mu)
            if self.n_restarts > 0:
                for init in np.random.uniform(self.bound[0], self.bound[1], size=self.n_restarts):
                    _mu = scpopt.minimize(LOOCV, np.array((init,)), method="L-BFGS-B", bounds=(self.bound,)).x[0]
                    _target = LOOCV(_mu)
                    if _target < target: self.mu, target = _mu, _target
            self.mu = np.power(10, self.mu)

            # todo: Calculate coefficients
            self.r_A_inv = np.matmul(np.matmul(np.matmul(G, np.diag(1.0 / (1.0 + n * self.mu * delta))), G_inv), self.A_inv)
            # self.r_A_inv = scipy.linalg.inv(self.A + n * self.mu * Sigma)
            self.c = np.matmul(self.r_A_inv, rhs)
            self.updated = True
            self.update_times += 1

    def predict(self, x, level=0.95):
        pred = {}
        self.fit()

        _x = to_unit_box(torch.atleast_2d(x), self.lb, self.ub)
        ds = scpspatial.distance.cdist(_x, self._X)
        mean = np.dot(self.kernel.eval(ds), self.c[:self.num_pts]) + np.dot(
            self.tail.eval(_x), self.c[self.num_pts:]
        )
        pred["mean"] = torch.tensor(mean, **tkwargs)

        phi0 = self.kernel.eval(np.zeros((_x.shape[0], 1)))
        u = np.hstack((self.kernel.eval(ds), self.tail.eval(_x)))
        power_value = np.subtract(phi0, np.matmul(np.matmul(u, self.r_A_inv), u.T).diagonal().reshape((_x.shape[0], 1)))
        rho = np.percentile([self.c[i] ** 2 / self.r_A_inv[i, i] for i in range(self._X.shape[0])], q=level * 100)
        pred["lowerbound"] = torch.tensor(mean - np.sqrt(np.abs(power_value * rho)), **tkwargs)
        pred["upperbound"] = torch.tensor(mean + np.sqrt(np.abs(power_value * rho)), **tkwargs)

        return pred



class HeteroRBFModel(Surrogate):
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

        # strategy = "N": no updating strategy on matrix inversion (small error but high computational cost);
        # strategy = "S": update matrix inversion with a single point each time;
        # strategy = "M": update matrix inversion with multiple points each time.
        self.strategy = options.get("strategy", "N")
        self.bound = options.get("bound", (-6.0, -1.0))  # Will be transformed to 10^bound when tuning regularization parameter
        self.n_restarts = options.get("n_restarts", 0)
        self.maxiter = options.get("maxiter", 2)
        self.moment_order = options.get("moment_order", 1.0)

        self.RBF_AUX = HomoRBFModel(ndim=self.ndim, lb=self.lb, ub=self.ub, strategy="M")

    def reset(self):
        super().reset()

        self.L = None
        self.U = None
        self.piv = None
        self.c = None

        self.mu = None
        self.noise_std = None
        self.update_times = 0

    def fit(self):
        if not self.updated:
            n = self.num_pts
            ntail = self.tail.ndim_tail
            self._X = to_unit_box(self.X, self.lb, self.ub).numpy()

            X = self._X[0:n, :]
            Phi = self.kernel.eval(scpspatial.distance.cdist(X, X)) + self.eta * np.eye(n)
            Pi = self.tail.eval(X)

            # todo: Compute (or Update) A_inv, i.e., the matrix inversion of A
            if self.strategy == "N" or self.update_times % 50 == 0:
                assert self.num_pts >= ntail
                A1 = np.hstack((Phi, Pi))
                A2 = np.hstack((Pi.T, np.zeros((ntail, ntail))))
                self.A = np.vstack((A1, A2))
                self.A_inv = scplinalg.inv(self.A)
                # print(np.allclose(np.matmul(self.A, self.A_inv), np.eye(self.A.shape[0])))
                # print(np.amax(np.abs(np.matmul(self.A, self.A_inv) - np.eye(self.A.shape[0]))))
                # print(np.allclose(np.matmul(self.A, scplinalg.inv(self.A)), np.eye(self.A.shape[0])))
                # print(np.amax(np.abs(np.matmul(self.A, scplinalg.inv(self.A)) - np.eye(self.A.shape[0]))))
            else:
                k = self.c.shape[0] - ntail
                numnew = n - k if self.strategy == "M" else 1
                while k < n:
                    A1 = np.hstack((Phi[:k + numnew, :k + numnew], Pi[:k + numnew, :]))
                    A2 = np.hstack((Pi[:k + numnew, :].T, np.zeros((ntail, ntail))))
                    self.A = np.vstack((A1, A2))

                    # Construct perturbation matrix
                    _P = np.zeros((numnew + ntail, numnew + ntail))
                    for i in range(ntail): _P[i, i + numnew] = 1
                    for i in range(ntail, numnew + ntail): _P[i, i - ntail] = 1

                    P = np.vstack((
                        np.hstack((np.eye(k), np.zeros((k, numnew + ntail)))),
                        np.hstack((np.zeros((numnew + ntail, k)), _P))
                    ))

                    _A = np.matmul(np.matmul(P, self.A), P.T)
                    B = _A[:k + ntail, k + ntail:]
                    C = _A[k + ntail:, :k + ntail]
                    D = _A[k + ntail:, k + ntail:]

                    # Block matriix inversion
                    CA_inv = np.matmul(C, self.A_inv)
                    A_invB = np.matmul(self.A_inv, B)
                    U = scplinalg.inv(np.subtract(D, np.matmul(CA_inv, B)))

                    _A_inv = np.vstack((
                        np.hstack((
                            np.add(self.A_inv, np.matmul(np.matmul(A_invB, U), CA_inv)),
                            -np.matmul(A_invB, U)
                        )),
                        np.hstack((-np.matmul(U, CA_inv), U)),
                    ))
                    self.A_inv = np.matmul(np.matmul(P.T, _A_inv), P)
                    k += numnew

            # todo: Regularization parameter estimation

            # todo: 1) Consider the unknown noise level as part of the regularization parameter;
            #       3) Ensure that self.A_inv @ Sigma has real eigen values and vectors by adding a small eta;

            self.noise_std = np.ones(n)

            for iter in range(self.maxiter):
                Sigma = np.vstack((
                    np.hstack((np.diag(self.noise_std ** 2), np.zeros((n, ntail)))),
                    np.hstack((np.zeros((ntail, n)), np.eye(ntail) * self.eta))
                ))

                delta, G = scplinalg.eig(np.matmul(self.A_inv, Sigma))
                delta, G, G_inv = np.real(delta), np.real(G), np.real(scplinalg.inv(G))

                fX = self.fX.numpy().copy()
                rhs = np.vstack((fX, np.zeros((ntail, 1))))

                pre_alpha = np.matmul(G_inv, self.A_inv)
                def LOOCV(mu):
                    mu = np.atleast_1d(np.power(10, mu))
                    _r_A_inv = np.matmul(np.matmul(G, np.diag(1.0 / (1.0 + n * mu[0] * delta))), pre_alpha)
                    _c = np.matmul(_r_A_inv, rhs)
                    f = sum([np.abs(_c[i, 0]) / self.noise_std[i] / np.abs(_r_A_inv[i, i]) for i in range(n)]) / n
                    return f

                self.mu = np.random.uniform(self.bound[0], self.bound[1]) if self.mu is None else np.log10(self.mu)
                self.mu = scpopt.minimize(LOOCV, np.array((self.mu,)), method="L-BFGS-B", bounds=(self.bound,)).x[0]
                target = LOOCV(self.mu)
                if self.n_restarts > 0:
                    for init in np.random.uniform(self.bound[0], self.bound[1], size=self.n_restarts):
                        _mu = scpopt.minimize(LOOCV, np.array((init,)), method="L-BFGS-B", bounds=(self.bound,)).x[0]
                        _target = LOOCV(_mu)
                        if _target < target: self.mu, target = _mu, _target
                self.mu = np.power(10, self.mu)

                # todo: Calculate coefficients
                self.r_A_inv = np.matmul(np.matmul(np.matmul(G, np.diag(1.0 / (1.0 + n * self.mu * delta))), G_inv), self.A_inv)
                # self.r_A_inv = scipy.linalg.inv(self.A + n * self.mu * Sigma)
                self.c = np.matmul(self.r_A_inv, rhs)

                if iter == 0:
                    # todo: Improved Most Likely Improvement
                    ds = scpspatial.distance.cdist(self._X, self._X)
                    fX_mean = np.dot(self.kernel.eval(ds), self.c[:self.num_pts]) + np.dot(
                        self.tail.eval(self._X), self.c[self.num_pts:]
                    )

                    z = np.power(np.abs(fX_mean - self.fX), self.moment_order)
                    self.RBF_AUX.reset()
                    self.RBF_AUX.add_points(xx=self.X, fx=z)
                    z_mean = self.RBF_AUX.predict(self.X).reshape((self.X.shape[0],))
                    correction_factor = np.power(np.pi, 0.5) / (
                                np.power(2.0, self.moment_order / 2.0) * scpspecial.gamma((self.moment_order + 1) / 2.0))
                    corrected_mean = np.maximum(z_mean * correction_factor, np.zeros(len(z_mean)))
                    self.noise_std = np.power(corrected_mean, 1.0 / self.moment_order)

                    # print("Noise estimation = {}".format(self.noise_std))

            self.updated = True
            self.update_times += 1

    def predict(self, x, level=0.95):
        pred = {}
        self.fit()

        _x = to_unit_box(torch.atleast_2d(x), self.lb, self.ub)
        ds = scpspatial.distance.cdist(_x, self._X)
        mean = np.dot(self.kernel.eval(ds), self.c[:self.num_pts]) + np.dot(
            self.tail.eval(_x), self.c[self.num_pts:]
        )
        pred["mean"] = torch.tensor(mean, **tkwargs)

        phi0 = self.kernel.eval(np.zeros((_x.shape[0], 1)))
        u = np.hstack((self.kernel.eval(ds), self.tail.eval(_x)))
        power_value = np.subtract(phi0, np.matmul(np.matmul(u, self.r_A_inv), u.T).diagonal().reshape((_x.shape[0], 1)))
        rho = np.percentile([self.c[i] ** 2 / self.r_A_inv[i, i] for i in range(self._X.shape[0])], q=level * 100)
        pred["lowerbound"] = torch.tensor(mean - np.sqrt(np.abs(power_value * rho)), **tkwargs)
        pred["upperbound"] = torch.tensor(mean + np.sqrt(np.abs(power_value * rho)), **tkwargs)

        return pred