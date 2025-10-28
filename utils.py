import argparse
import random

import numpy as np

MEASURES = ["synergy", "causation", "redundancy", "integrated", "emergence"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_gen", type=int, default=100)
    parser.add_argument("--kf", type=float, default=1e-2)
    parser.add_argument("--kb", type=float, default=1e-5)
    parser.add_argument("--A", type=float, default=-4.0)
    parser.add_argument("--sigma", type=float, default=4.0)
    return parser.parse_args()


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
