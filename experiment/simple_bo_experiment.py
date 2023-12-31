import torch

from BayesianOpt.problem import Rastrigin, Branin
from BayesianOpt.model import GaussianProcess
from BayesianOpt.design import QuasiMCDesign
from BayesianOpt.acquisition import UCB
from BayesianOpt.base import SurrogateOptimization


def run_experiments():
    N_TRIALS = 30
    N_ITERATIONS = 100

    for prob in [Branin()]:
        for trial_id in range(1, N_TRIALS + 1):
            torch.manual_seed(trial_id)

            run_single_trial(prob=prob, n_iterations=N_ITERATIONS, trial_id=trial_id, random_seed=trial_id)


def run_single_trial(prob, n_iterations, trial_id, random_seed):
    model = GaussianProcess(
        ndim=prob.ndim,
        lb=prob.lb,
        ub=prob.ub,
        interpolant=False,
    )

    design = QuasiMCDesign(
        ndim=prob.ndim,
        lb=prob.lb,
        ub=prob.ub,
        random_state=random_seed,
    )

    acquisition = UCB(
        ndim=prob.ndim,
        lb=prob.lb,
        ub=prob.ub,
    )

    algorithm = SurrogateOptimization(
        prob=prob,
        model=model,
        design=design,
        acquisition=acquisition,
    )

    algorithm.run(
        T=n_iterations,
        npts=1,
        maxiter=15000,
        n_restart=10,
        plot_progress=False,
        print_progress=True,
    )
    algorithm.save_results(root="../test_result", trial_id=trial_id)


if __name__ == "__main__":
    run_experiments()