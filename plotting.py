import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import MEASURES


def load_data(directory="output"):
    data = None
    for file in os.listdir(directory):
        if file.endswith("txt"):
            try:
                d = pd.read_csv(os.path.join(directory, file), sep=";")
            except pd.errors.EmptyDataError:
                continue
            d["seed"] = int(file.split(".")[0])
            if data is None:
                data = d.copy()
            else:
                data = pd.concat([data, d], axis=0)
    return data


def _compositional_similarity(v_chi, v_delta):
    v1 = v_chi / (v_chi.sum() if v_chi.sum() > 0 else 1)
    v2 = v_delta / (v_delta.sum() if v_delta.sum() > 0 else 1)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def _get_similarity_matrix(data_x, data_y):
    carpet = np.zeros((data_x["i"].max() + 1, data_y["i"].max() + 1))
    for (i,), traj_x in data_x.groupby(["i"]):
        v_chi = np.array([int(c) for c in traj_x["n"].item().split("/")])
        for (j,), traj_y in data_y.groupby(["i"]):
            print(i, j)
            if i <= j:
                continue
            s = _compositional_similarity(v_chi=v_chi,
                                          v_delta=np.array([int(c) for c in traj_y["n"].item().split("/")]))
            carpet[i, j] = s
            carpet[j, i] = s
    return carpet


def _get_thresholded_matrix(similarity, threshold=0.9):
    mask = similarity >= threshold
    consecutive_pairs = []
    for i in range(similarity.shape[0] - 1):
        for j in range(similarity.shape[1] - 1):
            if mask[i, j] and mask[i + 1, j + 1]:
                consecutive_pairs.append(((i, j), (i + 1, j + 1)))
    consecutive_matrix = np.zeros_like(similarity, dtype=int)
    for (i, j), (ip1, jp1) in consecutive_pairs:
        consecutive_matrix[i, j] = 1
        consecutive_matrix[ip1, jp1] = 1
    return consecutive_matrix


def carpet_plot(data, x, y):
    data_x = data[data["seed"] == x]
    data_y = data[data["seed"] == y]
    fig, axes = plt.subplots(figsize=(20, 5), nrows=1, ncols=2)

    carpet = _get_similarity_matrix(data_x=data_x,
                                    data_y=data_y)
    im = axes[0].imshow(carpet)
    axes[0].set_xlabel("generation", fontsize=15)
    axes[0].set_ylabel("generation", fontsize=15)
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.ax.set_title("compositional\nsimilarity")

    consecutive_matrix = _get_thresholded_matrix(similarity=carpet)
    axes[1].imshow(consecutive_matrix)

    plt.savefig(os.path.join("figures", f"carpet_{x}_{y}.png"))
    plt.close()


def plot_info():
    df = pd.read_csv("info.txt", sep=";")
    fig, axes = plt.subplots(figsize=(8, 5 * len(MEASURES)), nrows=len(MEASURES), ncols=1)
    for ax, measure in zip(axes, MEASURES):
        data = np.array(df[measure].str.split('/', expand=True), dtype=float)
        median = np.median(data, axis=0)
        ax.plot(median)
        err = np.std(data, axis=0)
        ax.fill_between(np.arange(len(median)), median - err, median + err, alpha=0.25)
        ax.set_title(measure, fontsize=15)
        ax.set_xlabel("simulation time", fontsize=10)
        ax.set_ylabel("nats", fontsize=10)
    plt.savefig("figures/info.png")
    plt.close()


if __name__ == "__main__":
    # d = load_data()
    # d = d[d["i"] < 200]
    # carpet_plot(data=d,
    #             x=0,
    #             y=1)
    plot_info()
