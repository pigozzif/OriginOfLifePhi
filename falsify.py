import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d

from algorithms import GeneticAlgorithm
from utils import parse_args, set_seed

if __name__ == "__main__":
    args = parse_args()
    set_seed(s=args.seed)
    data = loadmat(os.path.join(os.getcwd(), "GARD-model", "Matlab", "SourceCode",
                                "Doron_Lancet_GARD_Next_generation", "GARD_v10", "data",
                                f"GARD_run_seed_{args.seed:03d}.mat"))
    trace = data["o"]["history"][0, 0].astype(np.int32)
    # print(np.sum(trace, axis=0).shape)
    # print((np.sum(trace, axis=0) <= 2).shape)
    # print(np.any(np.sum(trace, axis=0) < 1))
    # plt.plot(uniform_filter1d(np.sum(trace, axis=0), size=25))
    # plt.show()
    solver = GeneticAlgorithm(trace=trace, maximize=True, n_gen=100)
    for sol, fitness in solver.learn():
        print(fitness)
