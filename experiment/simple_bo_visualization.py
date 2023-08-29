import matplotlib.pyplot as plt

from BayesianOpt.utils import DataHandler
from BayesianOpt.problem import Branin

algos = [{
    "name": "SurrogateOptimization",
}]

prob = Branin(ndim=2)

handler = DataHandler(
    root="../test_result",
    algos=algos,
    prob=prob,
)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

handler.plot_average(ax=ax)

plt.savefig("simple_bo_result.pdf")
plt.show()