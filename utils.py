import argparse
import random

import numpy as np

MEASURES = ["synergy", "causation", "redundancy", "integrated", "emergence"]
DESCRIPTORS = ["std",
               "trend",
               "monotonicity",
               "flatness",
               "gini",
               "max.peaks.number",
               "max.peaks.distance.mean",
               "max.peaks.distance.std",
               "min.peaks.number",
               "min.peaks.distance.mean",
               "min.peaks.distance.std",
               "all.peaks.number",
               "all.peaks.distance.mean",
               "all.peaks.distance.std",
               "max.peaks.val.mean",
               "max.peaks.val.std",
               "min.peaks.val.mean",
               "min.peaks.val.std",
               "all.peaks.val.mean",
               "all.peaks.val.std",
               "max.min.diff.mean",
               "is_flat"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_gen", type=int, default=100)
    parser.add_argument("--kf", type=float, default=1e-2)
    parser.add_argument("--kb", type=float, default=1e-5)
    parser.add_argument("--A", type=float, default=-4.0)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--method", type=str, default="diff")
    parser.add_argument("--p", type=float, default=0.5)
    return parser.parse_args()


def set_seed(s):
    random.seed(s)
    np.random.seed(s)


def generate_sequences(sequence, window_size, step=1):
    n_steps = (len(sequence) - window_size) // step + 1
    windows = [sequence[i: i + window_size] for i in range(0, n_steps * step, step)]
    return np.array(windows)


def get_info_array(df, measure, fill_value=np.nan):
    data = np.full((len(df[measure]), np.max([len(arr) for arr in df[measure].values()])),
                   fill_value=fill_value)
    for i, arr in df[measure].items():
        data[i, :len(arr)] = arr
    return data
