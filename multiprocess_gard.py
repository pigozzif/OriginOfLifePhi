import os
import sys
from multiprocessing import Pool


if __name__ == "__main__":
    worker_id = int(sys.argv[1])
    n_seeds = 1000
    with Pool(os.cpu_count()) as pool:
        pool.map(os.system, [f"python gard.py --seed={i}" for i in range(n_seeds * worker_id,
                                                                         n_seeds * (worker_id + 1),
                                                                         1)])
