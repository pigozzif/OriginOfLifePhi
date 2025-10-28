import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import uniform_filter1d
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from utils import MEASURES


def load_data(directory="output"):
    data = None
    for file in os.listdir(directory):
        if file.endswith("txt"):
            try:
                curr_d = pd.read_csv(os.path.join(directory, file), sep=";")
            except pd.errors.EmptyDataError:
                continue
            curr_d["seed"] = int(file.split(".")[0])
            if data is None:
                data = curr_d.copy()
            else:
                data = pd.concat([data, curr_d], axis=0)
    data["n"] = data.apply(lambda row: np.array([int(c) for c in row["n"].split("/")]), axis=1)
    data["n"] = data.apply(lambda row: row["n"] / row["n"].sum(), axis=1)
    return data


def _compositional_similarity(v_chi, v_delta):
    v1 = v_chi / (v_chi.sum() if v_chi.sum() > 0 else 1)
    v2 = v_delta / (v_delta.sum() if v_delta.sum() > 0 else 1)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def _get_similarity_matrix(data_x, data_y):
    carpet = np.zeros((data_x["i"].max() + 1, data_y["i"].max() + 1))
    for (i,), traj_x in data_x.groupby(["i"]):
        for (j,), traj_y in data_y.groupby(["i"]):
            print(i, j)
            if i <= j:
                continue
            try:
                s = _compositional_similarity(v_chi=traj_x["n"].item(),
                                              v_delta=traj_y["n"].item())
            except:
                print(traj_x["n"], traj_y["n"])
                raise
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


def carpet_plot(args):  # data, x, y):
    data, x = args
    data_x = data[data["seed"] == x]
    fig, axes = plt.subplots(figsize=(20, 5), nrows=1, ncols=1)

    arr = np.stack(data_x["n"].to_numpy())
    carpet = cosine_similarity(arr)#_get_similarity_matrix(data_x=data_x,
                                #    data_y=data_x)
    im = axes.imshow(carpet)
    axes.set_xlabel("generation", fontsize=15)
    axes.set_ylabel("generation", fontsize=15)
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.ax.set_title("compositional\nsimilarity")

    # consecutive_matrix = _get_thresholded_matrix(similarity=carpet)
    # axes[1].imshow(consecutive_matrix)

    plt.savefig(os.path.join("figures", "carpets", f"carpet_{x}.png"))
    plt.close()


def plot_molar_fractions(data):
    n_cols = len(data["seed"].unique())
    fig, axes = plt.subplots(figsize=(8 * n_cols, 5), nrows=1, ncols=n_cols)
    for ax, ((seed,), traj) in zip(axes, data.groupby(["seed"])):
        arr = np.stack(traj["n"].to_numpy())
        fractions = arr / arr.sum(axis=1)[:, np.newaxis]
        ax.plot(uniform_filter1d(fractions, size=100, axis=0), alpha=0.25)
        ax.set_title(str(seed), fontsize=15)
    plt.savefig("figures/fractions.png")
    plt.close()


def plot_asymptotic_sim(data):
    n_cols = len(data["seed"].unique())
    fig, axes = plt.subplots(figsize=(8 * n_cols, 5), nrows=1, ncols=n_cols)
    for ax, ((seed,), traj) in zip(axes, data.groupby(["seed"])):
        arr = np.stack(traj["n"].to_numpy())
        target = arr[-1]
        sim = np.array([_compositional_similarity(v_chi=target,
                                                  v_delta=comp) for comp in arr])
        ax.plot(uniform_filter1d(sim, size=1, axis=0))
        ax.set_title(str(seed), fontsize=15)
        ax.set_ylim(0.0, 1.0)
    plt.savefig("figures/sim.png")
    plt.close()


def plot_average_sim(data):
    sims = np.zeros((len(data["seed"].unique()), data[""]))
    for (seed,), traj in data.groupby(["seed"]):
        arr = np.stack(traj["n"].to_numpy())
        target = arr[-1]
        sims[seed] = np.array([_compositional_similarity(v_chi=target,
                                                         v_delta=comp) for comp in arr])
    median = np.median(sims, axis=0)
    plt.plot(median)
    err = np.std(sims, axis=0)
    plt.fill_between(np.arange(len(median)), median - err, median + err, alpha=0.25)
    plt.savefig("figures/avg.sim.png")
    plt.close()


def choose_pca_components(X):
    pca = PCA().fit(X)
    evr = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(evr) + 1), evr, "o-")
    plt.axhline(0.9, color='gray', ls='--')
    plt.xlabel("Components")
    plt.ylabel("Cumulative explained variance")
    plt.show()
    k_opt = np.argmax(evr > 0.9) + 1
    return k_opt


def plot_pca_evr():
    df = pd.read_csv("info.txt", sep=";")
    fig, axes = plt.subplots(figsize=(8, 5 * len(MEASURES)), nrows=len(MEASURES), ncols=1)
    for ax, measure in zip(axes, MEASURES):
        data = np.array(df[measure].str.split('/', expand=True), dtype=float)
        pca = PCA().fit(data)
        evr = np.cumsum(pca.explained_variance_ratio_)
        ax.plot(range(1, len(evr) + 1), evr, "o-")
        ax.axhline(0.9, color="gray", ls='--')
        ax.set_xlabel("Components")
        ax.set_ylabel("Cumulative explained variance")
        ax.set_title(measure, fontsize=15)
    plt.savefig("figures/pca.png")
    plt.close()


def plot_info():
    df = pd.read_csv("info.txt", sep=";")
    # df = df[df["seed"] == np.random.randint(0, df["seed"].max())]
    fig, axes = plt.subplots(figsize=(8, 5 * len(MEASURES)), nrows=len(MEASURES), ncols=1)
    for ax, measure in zip(axes, MEASURES):
        data = np.array(df[measure].str.split('/', expand=True), dtype=float)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        # data = StandardScaler().fit_transform(data)
        data = uniform_filter1d(data, size=1, axis=0)
        median = np.nanmedian(data, axis=0)
        print(np.min(median), np.max(median))
        # median = uniform_filter1d(median, size=1)
        # median = StandardScaler().fit_transform(median.reshape(-1, 1))[:, 0]
        ax.plot(median)
        x = np.arange(len(median))  # StandardScaler().fit_transform(np.arange(len(median)).reshape(-1, 1))[:, 0]
        beta_1, beta_0, rvalue, pvalue, _ = linregress(x, median)
        y_hat = beta_0 + (beta_1 * x)
        ax.plot(x, y_hat, color="red", label=f"r={round(rvalue, 3)}, p={pvalue}")
        ax.legend()
        err = np.std(data, axis=0)
        ax.fill_between(x, median - err, median + err, alpha=0.25)
        ax.set_title(measure, fontsize=15)
        ax.set_xlabel("simulation time", fontsize=10)
        ax.set_ylabel("nats", fontsize=10)
        # ax.set_ylim(-1.0, 1.0)
    plt.savefig("figures/info.png")
    plt.close()


def plot_diagnostics(data, n=1000, exp_name="figures"):
    arr = np.stack(data.sample(n)["n"].to_numpy())
    sim = cosine_similarity(arr)
    plt.imshow(sim, cmap="viridis")
    plt.colorbar()
    plt.savefig(f"{exp_name}/sim.png")
    plt.close()

    arr = normalize(arr, norm="l2")
    scores = []
    for k in range(2, 20):
        km = KMeans(n_clusters=k, random_state=0).fit(arr)
        scores.append(silhouette_score(arr, km.labels_))

    plt.plot(range(2, 20), scores, 'o-')
    plt.xlabel("k")
    plt.ylabel("Silhouette score")
    plt.savefig(f"{exp_name}/scores.png")
    plt.close()


if __name__ == "__main__":
    d = load_data()
    # d = d[d["i"] < 500]
    # plot_info()
    # plot_pca_evr()
    plot_diagnostics(data=d)
    n_seeds = 10
    d = d[d["seed"] < n_seeds]
    # d = d[d["is_parent"]]
    plot_molar_fractions(data=d)
    plot_asymptotic_sim(data=d)
    for i in range(n_seeds):
        carpet_plot((d, i))
