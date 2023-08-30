import glob
import torch
import numpy as np


class DataHandler:
    def __init__(self, root, algos, prob):
        self.root = root
        self.algos = algos
        self.prob = prob

        self.get_solutions()
        self.analyze_solutions()

    def get_solutions(self):
        for algo in self.algos:
            filenames = sorted(glob.iglob(f"{self.root}/{algo['name']}/{self.prob.name}/{self.prob.ndim}/*.txt"))
            algo["nruns"] = len(filenames)

            algo["X"] = []
            algo["fX"] = []
            algo["fX_observe"] = []
            for filename in filenames:
                record = np.loadtxt(filename)
                algo["X"].append(record[:, :self.prob.ndim])
                algo["fX_observe"].append(record[:, self.prob.ndim])
                algo["fX"].append(self.prob._eval(torch.tensor(record[:, :self.prob.ndim])).numpy())


    def analyze_solutions(self):
        for algo in self.algos:
            algo["fXbest"] = []
            for fX in algo["fX"]:
                algo["fXbest"].append([np.float_("inf")])
                for _fX in fX:
                    algo["fXbest"][-1].append(min(algo["fXbest"][-1][-1], _fX))
                algo["fXbest"][-1].pop(0)

            algo["fXbest_avg"] = np.mean(algo["fXbest"], axis=0)
            algo["fXbest_std"] = np.std(algo["fXbest"], axis=0)

    def plot_average(self, ax, start=None, end=None, show_CI=False):
        for algo in self.algos:
            _start = start if start is not None and 0 <= start <= len(algo["fXbest_avg"]) - 1 else 0
            _end = end if end is not None and start <= end <= len(algo["fXbest_avg"]) - 1 else len(algo["fXbest_avg"]) - 1

            iter = np.arange(_start, _end + 1) + 1
            avg = algo["fXbest_avg"][_start:_end + 1]
            std = algo["fXbest_std"][_start:_end + 1]

            line = ax.plot(iter, avg, label=algo["name"])
            if show_CI:
                ax.fill_between(iter, avg, avg + 1.96 * std / np.sqrt(algo["nruns"]), color=line[0]._color, alpha=0.3)
                ax.fill_between(iter, avg, avg - 1.96 * std / np.sqrt(algo["nruns"]), color=line[0]._color, alpha=0.3)