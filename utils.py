import argparse
import random

import numpy as np

MEASURES = ["synergy", "causation", "redundancy", "integrated", "emergence"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_gen", type=int, default=5000)
    return parser.parse_args()


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
