import os
import sys
from multiprocessing import Pool

if __name__ == "__main__":
    # worker_id = int(sys.argv[1])
    n_seeds = 100
    with Pool(5) as pool:
        pool.map(os.system, [f"python rl.py --seed={i}" for i in range(n_seeds)])  # range(2, n_seeds * worker_id,
        #         n_seeds * (worker_id + 1),
        #         1)])
