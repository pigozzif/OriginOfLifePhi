import os

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from scipy.io import loadmat, savemat
from scipy.ndimage import uniform_filter1d, minimum_filter1d
from scipy.stats import linregress, spearmanr, kendalltau, pearsonr, zscore, wilcoxon, ttest_1samp
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import RidgeCV, Ridge, LogisticRegression, LinearRegression
from sklearn.manifold import TSNE, Isomap
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import silhouette_score, make_scorer, mean_squared_error, log_loss
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor, plot_tree, DecisionTreeClassifier
from sklearn.utils import resample
from sktime.distances import dtw_distance
from sktime.dists_kernels import dtw
from statsmodels.tsa.stattools import grangercausalitytests

from attention import AttentionRegressor, AttentionClassifier
from utils import MEASURES, set_seed, DESCRIPTORS, get_info_array

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']


def load_data(directory="output"):
    data = None
    for file in os.listdir(directory):
        if file.endswith("txt"):
            try:
                curr_d = pd.read_csv(os.path.join(directory, file), sep=";")
            except pd.errors.EmptyDataError:
                continue
            curr_d["seed"] = int(file.split(".")[-2])
            if "learning" in directory:
                curr_d["method"] = file.split(".")[1]
                curr_d["p"] = float(file.split(".")[0].replace(',', '.'))
            if data is None:
                data = curr_d.copy()
            else:
                data = pd.concat([data, curr_d], axis=0)
    # data["n"] = data.apply(lambda row: np.array([int(c) for c in row["n"].split("/")]), axis=1)
    # data["n"] = data.apply(lambda row: row["n"] / row["n"].sum(), axis=1)
    return data


def load_mat_data(path=os.path.join(os.getcwd(), "GARD-model", "Matlab", "SourceCode",
                                    "Doron_Lancet_GARD_Next_generation", "GARD_v10", "data")):
    data = None
    for num, file in enumerate(os.listdir(path)):
        if not file.endswith("mat"):
            continue
        curr_d = loadmat(os.path.join(path, file))
        trace = curr_d["o"]["trace"][0, 0]
        trace = trace.reshape(1, *trace.shape)
        if data is None:
            data = trace.copy()
        else:
            data = np.concatenate([data, trace.copy()], axis=0)
    return data


def load_all_mat_data(path=os.path.join(os.getcwd(), "GARD-model", "Matlab", "SourceCode",
                                        "Doron_Lancet_GARD_Next_generation", "GARD_v10", "data"),
                      attribute="history"):
    data = []
    for num, file in enumerate(os.listdir(path)):
        if not file.endswith("mat"):
            continue
        curr_d = loadmat(os.path.join(path, file))
        trace = curr_d["o"][attribute][0, 0]
        data.append(trace)
    return data


def load_mat_types(path=os.path.join(os.getcwd(), "GARD-model", "Matlab", "SourceCode",
                                     "Doron_Lancet_GARD_Next_generation", "GARD_v10", "data"),
                   binary=False):
    data = {}
    for num, file in enumerate(os.listdir(path)):
        if not file.endswith("mat"):
            continue
        curr_d = loadmat(os.path.join(path, file))
        tags = curr_d["o"]["tags"][0, 0]
        if binary:
            tags = np.where(tags != 0, 1, 0)
        data[num] = tags
    return data


def load_descriptors(file_name="features.txt"):
    df = pd.read_csv(file_name, sep=";")
    return df


def read_info(directory="output"):
    info = {}
    for measure in MEASURES:
        root = os.path.join(directory, measure.lower())
        arrs = {int(file.split(".")[0]): np.load(os.path.join(root, file))
                for file in os.listdir(os.path.join(directory, measure.lower()))}
        # data = np.full((len(arrs), np.max([len(arr) for arr in arrs.values()])), np.nan)
        # for i, arr in arrs.items():
        #     data[i, :len(arr)] = arr
        info[measure] = arrs  # data
    data = {}
    for i in info["synergy"].keys():  # arr1, arr2 in zip(info["synergy"], info["causation"]):
        arr1 = info["synergy"][i]
        arr2 = info["causation"][i]
        # min_length = min(len(arr1[~np.isnan(arr1)]), len(arr2[~np.isnan(arr2)]))
        emergence = np.nansum([arr1, arr2], axis=0).flatten()
        # emergence[min_length:] = np.nan
        # data.append(emergence)
        data[i] = emergence
    info["emergence"] = data  # np.array(data)
    # print(info["emergence"].shape, info["synergy"].shape, info["causation"].shape)
    return info


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
    carpet = cosine_similarity(arr)  #_get_similarity_matrix(data_x=data_x,
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


def plot_info(exp="output_fluxes"):
    df = read_info(directory=exp)
    tags = load_mat_types(binary=True)
    # seed = np.random.randint(0, df["seed"].max())
    # print(seed)
    # df = df[df["seed"] == seed]
    measures = df.keys()
    fig, axes = plt.subplots(figsize=(8, 5 * (len(measures) + 1)), nrows=len(measures) + 1, ncols=1)
    run = np.random.randint(0, len(df["synergy"]))
    for ax, measure in zip(axes, measures):
        # data = np.array(df[measure].str.split('/', expand=True), dtype=float)
        data = df[measure][run].reshape(1, -1)
        # data = np.full((len(df[measure]), np.max([len(arr) for arr in df[measure].values()])), np.nan)
        # for i, arr in df[measure].items():
        #     data[i, :len(arr)] = arr
        # data = np.array([np.array(d) for d in data.values()])
        # data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        # data = StandardScaler().fit_transform(data)
        # data = uniform_filter1d(data, size=100, axis=0)
        # data = zscore(data, axis=1, nan_policy="omit")
        median = np.nanmedian(data, axis=0)
        # print(np.min(median), np.max(median))
        from scipy.signal import find_peaks
        # max_peaks = find_peaks(median.flatten())[0]
        # min_peaks = find_peaks(-median.flatten())[0]
        median = uniform_filter1d(median, size=10)
        # median = StandardScaler().fit_transform(median.reshape(-1, 1))[:, 0]
        ax.plot(median)
        # ax.scatter(max_peaks, median[max_peaks], color="green")
        # ax.scatter(min_peaks, median[min_peaks], color="green")
        x = np.arange(len(median))  # StandardScaler().fit_transform(np.arange(len(median)).reshape(-1, 1))[:, 0]
        beta_1, beta_0, rvalue, pvalue, _ = linregress(x, median)
        y_hat = beta_0 + (beta_1 * x)
        ax.plot(x, y_hat, color="red", label=f"r={round(rvalue, 3)}, p={pvalue}")
        ax.legend()
        err = np.nanstd(data, axis=0)
        ax.fill_between(x, median - err, median + err, alpha=0.25)
        ax.set_title(measure, fontsize=15)
        ax.set_xlabel("molecular step", fontsize=10)
        ax.set_ylabel("nats", fontsize=10)
        # ax.set_ylim(-1.0, 1.0)
    attractors = tags[run].flatten()
    attractors = minimum_filter1d(attractors, size=50)
    axes[-1].plot(attractors)
    plt.savefig("figures/info_tags2.png")
    plt.close()


def plot_samples(exp="output", measure="emergence", n=3):
    df = read_info(directory=exp)
    fig, axes = plt.subplots(figsize=(8 * (n + 1), 5), nrows=1, ncols=n + 1)
    runs = np.random.randint(0, len(df[measure]), size=n)
    data = get_info_array(df=df, measure=measure)
    median = np.nanmedian(data, axis=0)
    median = uniform_filter1d(median, size=1)
    axes[0].plot(median)
    x = np.arange(len(median))
    beta_1, beta_0, rvalue, pvalue, _ = linregress(x, median)
    y_hat = beta_0 + (beta_1 * x)
    axes[0].plot(x, y_hat, color="red", label=f"p={round(pvalue, 4)} > 0.05")
    axes[0].legend()
    err = np.nanstd(data, axis=0)
    axes[0].fill_between(x, median - err, median + err, alpha=0.25)

    for ax, run in zip(axes[1:], runs):
        d = df[measure][run]
        ax.plot(d)

    for ax in axes:
        ax.set_xlabel("molecular step", fontsize=15)
        ax.set_ylabel("Φ (nats)", fontsize=15)
    axes[0].set_title("A) Median±std of Φ over 100 runs", weight="bold", fontsize=20)
    for ax, title in zip(axes[1:], ["B", "C", "D", "E", "F"]):
        ax.set_title(f"{title}) Sample run", weight="bold", fontsize=20)

    fig.tight_layout()
    plt.savefig("figures/figure_1.png")
    plt.close()


def plot_info_tags(exp="output", n=10):
    df = read_info(directory=exp)
    tags = load_mat_types(binary=True)
    fig, axes = plt.subplots(figsize=(8 * n, 10), nrows=2, ncols=n)
    runs = np.random.randint(0, len(df["synergy"]), size=n)
    for i, run in enumerate(runs):
        data = df["emergence"][run].reshape(1, -1)
        median = np.nanmedian(data, axis=0)
        median = uniform_filter1d(median, size=10)
        axes[0][i].plot(median)
        x = np.arange(len(median))
        # beta_1, beta_0, rvalue, pvalue, _ = linregress(x, median)
        # y_hat = beta_0 + (beta_1 * x)
        # axes[0][i].plot(x, y_hat, color="red", label=f"r={round(rvalue, 3)}, p={pvalue}")
        # axes[0][i].legend()
        err = np.nanstd(data, axis=0)
        axes[0][i].fill_between(x, median - err, median + err, alpha=0.25)
        axes[0][i].set_title("emergence", fontsize=15)
        axes[0][i].set_xlabel("molecular step", fontsize=10)
        axes[0][i].set_ylabel("Φ (nats)", fontsize=10)
        attractors = tags[run].flatten()
        attractors = minimum_filter1d(attractors, size=50)
        axes[1][i].plot(attractors)
    plt.savefig("figures/tags_array.png")
    plt.close()


def granger_causality(exp="output", measure="emergence", maxlag=30):
    tags = load_mat_types(binary=True)
    df = read_info(directory=exp)
    data = df[measure]
    fig, axes = plt.subplots(figsize=(24, 5), nrows=1, ncols=2)
    n_runs = len(data)
    # for ax, measure in zip(axes, MEASURES):
    results = []
    for i, run in data.items():
        phi, c = run.flatten(), tags[i][:-1].flatten()
        if len(phi) != len(c):
            continue
        data_cf = np.column_stack([phi, c])  # C → Φ
        data_fc = np.column_stack([c, phi])  # Φ → C
        g_cf = grangercausalitytests(data_cf, maxlag=maxlag, verbose=False)
        g_fc = grangercausalitytests(data_fc, maxlag=maxlag, verbose=False)
        p_cf = min(g_cf[lag][0]["ssr_ftest"][1] for lag in range(1, maxlag + 1))
        p_fc = min(g_fc[lag][0]["ssr_ftest"][1] for lag in range(1, maxlag + 1))
        results.append({"run": i, "p_cf": p_cf, "p_fc": p_fc})

    df_res = pd.DataFrame(results)
    median_p_cf = df_res["p_cf"].median()
    median_p_fc = df_res["p_fc"].median()
    frac_cf_sig = np.mean(df_res["p_cf"] < 0.05)
    frac_fc_sig = np.mean(df_res["p_fc"] < 0.05)
    print(measure.upper())
    print(f"Composome→Φ: median p = {median_p_cf:.3g}, fraction sig = {frac_cf_sig:.2f}")
    print(f"Φ→Composome: median p = {median_p_fc:.3g}, fraction sig = {frac_fc_sig:.2f}")

    # axes[0].scatter(df_res["p_cf"], df_res["p_fc"],
    #                 color='k', alpha=0.7)
    # axes[0].hist2d(-np.log10(df_res["p_cf"]), -np.log10(df_res["p_fc"]), bins=25)
    # axes[0].axhline(0.05, color='r', linestyle='--')
    # axes[0].axvline(0.05, color='r', linestyle='--')
    # axes[0].set_xlabel("−log(p) for C → Φ")
    # axes[0].set_ylabel("−log(p) for Φ → C")
    # axes[0].set_title(measure)

    sig_cf = df_res["p_cf"] < 0.05  # C → Φ significant
    sig_fc = df_res["p_fc"] < 0.05  # Φ → C significant
    d = (sig_cf.sum() - sig_fc.sum()) / len(df_res)
    print(f"Directionality index D = {d:.2f}")
    bidirectional = sig_cf & sig_fc
    unidirectional_CF = sig_cf & ~sig_fc
    unidirectional_FC = sig_fc & ~sig_cf
    nonsig = ~(sig_cf | sig_fc)
    bidirectional_ratio = (sig_cf | sig_fc).mean()
    print(f"Directionality ratio: {bidirectional_ratio:.2f}")

    counts = {
        "Bidirectional": bidirectional.sum() / n_runs,
        "Composome→Φ only": unidirectional_CF.sum() / n_runs,
        "Φ→Composome only": unidirectional_FC.sum() / n_runs,
        "None": nonsig.sum() / n_runs
    }
    sns.barplot(
        x=["Granger causality", "None"],
        y=[(bidirectional.sum() + unidirectional_CF.sum() + unidirectional_FC.sum()) / n_runs, nonsig.sum() / n_runs],
        palette=["dimgray", "lightgray"],
        ax=axes[0]
    )
    axes[0].tick_params(axis="x", labelsize=15)
    sns.barplot(
        x=list(counts.keys()),
        y=list(counts.values()),
        palette=["gold", "tab:blue", "tab:orange", "lightgray"],
        ax=axes[1]
    )
    axes[1].tick_params(axis="x", labelsize=15)
    for ax in axes:
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f"{y * 100:g}")
        )
        ax.set_ylabel("%", fontsize=15)
    axes[0].set_title("A) % of runs with significant Granger causality\nbetween composomes and Φ, or not",
                      weight="bold", fontsize=20)
    axes[1].set_title("B) % of runs per causal direction",
                      weight="bold", fontsize=20)

    # axes[2].hist(df_res["p_cf"], bins=30, alpha=0.6, label="C→Φ", color="tab:blue")
    # axes[2].hist(df_res["p_fc"], bins=30, alpha=0.6, label="Φ→C", color="tab:orange")
    # axes[2].axvline(0.05, color='r', linestyle='--', label="p=0.05")
    # axes[2].legend()
    # axes[2].set_xlabel("p-value")
    # axes[2].set_ylabel("Count")
    # axes[2].set_title(measure)
    fig.tight_layout()
    plt.savefig("figures/figure_3.png")
    plt.close()


def granger_and_descriptors(exp="output", maxlag=25):
    tags = load_mat_types(binary=True)
    df = read_info(directory=exp)
    desc = pd.read_csv("features.txt", sep=";")
    # fig, axes = plt.subplots(figsize=(8, 5 * len(MEASURES)), nrows=len(MEASURES), ncols=1)
    for idx, measure in enumerate(MEASURES):
        # if measure != "emergence":
        #     continue
        data = df[measure]
        results = []
        for i, run in data.items():
            phi, c = run.flatten(), tags[i][:-1].flatten()
            if len(phi) != len(c):
                continue
            data_cf = np.column_stack([phi, c])  # C → Φ
            data_fc = np.column_stack([c, phi])  # Φ → C
            g_cf = grangercausalitytests(data_cf, maxlag=maxlag, verbose=False)
            g_fc = grangercausalitytests(data_fc, maxlag=maxlag, verbose=False)
            p_cf = min(g_cf[lag][0]["ssr_ftest"][1] for lag in range(1, maxlag + 1))
            p_fc = min(g_fc[lag][0]["ssr_ftest"][1] for lag in range(1, maxlag + 1))
            res = {"run": i, "p_cf": p_cf, "p_fc": p_fc}
            for d in DESCRIPTORS:
                res[d] = desc.loc[i, d].item()
            results.append(res)

        df_res = pd.DataFrame(results)
        sig_cf = df_res["p_cf"] < 0.05  # C → Φ significant
        sig_fc = df_res["p_fc"] < 0.05  # Φ → C significant
        bidirectional = sig_cf & sig_fc
        unidirectional_CF = sig_cf & ~sig_fc
        unidirectional_FC = sig_fc & ~sig_cf
        nonsig = ~(sig_cf | sig_fc)

        x, y = (df_res[DESCRIPTORS], bidirectional.astype(int) * 0 +
                unidirectional_CF.astype(int) * 1 +
                unidirectional_FC.astype(int) * 2 +
                nonsig.astype(int) * 3)
        x = TSNE(n_components=2).fit_transform(x)
        print(measure, silhouette_score(x, y))
        continue
        scores = []
        for k in range(2, 20):
            km = KMeans(n_clusters=k, random_state=42).fit(x)
            # labels = km.predict(x)
            # scores.append(silhouette_score(x, labels))
            scores.append(km.inertia_)
        plt.plot(scores)
        plt.show()
        # model = make_pipeline(
        #     StandardScaler(),
        #     RandomForestRegressor() #  MLPRegressor(hidden_layer_sizes=32,
        # activation="tanh",
        # solver="adam",
        # max_iter=1000,
        # random_state=42)
        # )
        # scores = cross_val_score(model, x, y, cv=5, scoring="r2")
        # print(f"R²: {scores.mean()} ± {scores.std()}")

        values = np.zeros((4, len(DESCRIPTORS)))
        for i, d in enumerate(DESCRIPTORS):
            for j, arr in enumerate([bidirectional, unidirectional_CF, unidirectional_FC, nonsig]):
                values[j, i] = df_res[arr][d].median()
        values = normalize(values)
        im = axes[idx].imshow(values)
        divider = make_axes_locatable(axes[idx])
        cax = divider.append_axes("right", size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")

    plt.savefig("figures/granger_desc.png")
    plt.close()


def plot_correlations(exp="output"):
    tags = load_mat_types(binary=True)
    df = read_info(directory=exp)
    fig, axes = plt.subplots(figsize=(8, 5 * len(MEASURES)), nrows=len(MEASURES), ncols=1)
    for ax, measure in zip(axes, MEASURES):
        data = df[measure]
        # data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        # data = StandardScaler().fit_transform(data)
        # data = normalize(data, axis=1)
        # data = zscore(data, axis=1, nan_policy='omit')
        pairs = []
        for method in [spearmanr, kendalltau, pearsonr]:
            corrs = []
            for s, run in data.items():
                phi, c = run.flatten(), tags[s][:-1].flatten()  # tags[s][:-1].flatten())
                if len(phi) != len(c):
                    continue
                r, p = method(phi, c)
                corrs.append(r)
                pairs.append((c, zscore(phi)))
            t_stat, p_value = wilcoxon(corrs)
            print(f"{measure}/{str(method)}: {p_value}")

        phi_comp = np.concatenate([run[s != 0] for s, run in pairs])
        phi_drift = np.concatenate([run[s == 0] for s, run in pairs])
        u, p = mannwhitneyu(_filter_outliers(phi_comp), _filter_outliers(phi_drift), alternative="greater")
        delta, magnitude = _cliffs_delta(phi_comp, phi_drift)
        print(f"Mann–Whitney U = {u:.2f}, p = {p:.3g}")
        print(f"Cliff's delta = {delta:.3f} ({magnitude} effect)")

        # ax.boxplot([phi_comp, phi_drift])
        ax.hist(phi_comp, alpha=0.6, color="tab:orange", label="comp")
        ax.hist(phi_drift, alpha=0.6, color="tab:blue", label="drift")
        ax.set_title(measure)
        ax.legend()
    plt.savefig("figures/boxplot_fluxes.png")
    plt.close()


def plot_mixed_effects(exp="output"):
    tags = load_mat_types(binary=True)
    info = read_info(directory=exp)
    for measure in MEASURES:
        records = []
        data = info[measure]
        for run_id, (s, run) in enumerate(data.items()):
            phi, c = run.flatten(), tags[s][:-1].flatten()
            if len(phi) != len(c):
                continue
            for ti, (pi, ci) in enumerate(zip(phi, c)):
                records.append({"run": run_id, "time": ti, "phi": pi, "c": ci})
        df = pd.DataFrame(records)
        df["phi"] = (df["phi"] - df["phi"].mean()) / df["phi"].std()
        df["c"] = (df["c"] - df["c"].mean()) / df["c"].std()
        df["time"] = (df["time"] - df["time"].mean()) / df["time"].std()

        import statsmodels.formula.api as smf
        model = smf.mixedlm("c ~ phi + time", data=df, groups=df["run"]).fit()
        print(f"{measure.upper()}: slope: {model.params['phi']} p-value: {model.pvalues['phi']}")


def plot_corr(exp="output"):
    tags = load_mat_types(binary=True)
    df = read_info(directory=exp)
    for measure in MEASURES:
        data = df[measure]
        # data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        # data = StandardScaler().fit_transform(data)
        # data = normalize(data, axis=1)
        # data = zscore(data, axis=1, nan_policy="omit")
        for method in [spearmanr, kendalltau, pearsonr]:
            agg_phi, rep_len = [], []
            corrs, ps = [], []
            for s, run in data.items():
                phi, c = run.flatten(), tags[s][:-1].flatten()  # tags[s][:-1].flatten()
                # phi = np.diff(phi)
                if len(phi) != len(c):
                    continue
                agg_phi.append(np.median(phi[c == 1]))
                rep_len.append(np.sum(c))
                r, p = method(phi, c)
                corrs.append(r)
                ps.append(p)
            t_stat, p_value = ttest_1samp(corrs, popmean=0.0, alternative="greater")
            ps, corrs = np.array(ps), np.array(corrs)
            print(
                f"{measure}/{str(method)}: {p_value} median corr: {np.median(corrs)} mean corr: {np.mean(corrs)} pos prop: {np.mean(corrs > 0)} pos prop & sig: {np.sum(ps[corrs > 0.0] < 0.05) / len(ps)} corr. w/ len{method(agg_phi, rep_len)}")

            if measure == "emergence" and method is spearmanr:
                fig, axes = plt.subplots(figsize=(16, 5), nrows=1, ncols=2)

                axes[0].hist(corrs, bins=20)
                axes[0].axvline(0.0, color="black", linestyle='--', linewidth=1,
                                label="$\\rho$=0")
                axes[0].axvline(np.mean(corrs), color='r', linestyle='--', linewidth=2,
                                label=f"mean $\\rho$={round(np.mean(corrs), 4)}")
                axes[0].set_xlim(-1.0, 1.0)
                axes[0].set_xlabel("Spearman's $\\rho$", fontsize=15)
                axes[0].set_ylabel("Count", fontsize=15)
                axes[0].set_title("A) Corr. coefficient", weight="bold", fontsize=15)
                axes[0].legend()

                sns.barplot(
                    x=[0, 1, 2, 3],
                    y=[np.sum(ps[corrs > 0.0] < 0.05) / len(ps),
                       np.sum(ps[corrs > 0.0] >= 0.05) / len(ps),
                       np.sum(ps[corrs <= 0.0] < 0.05) / len(ps),
                       np.sum(ps[corrs <= 0.0] >= 0.05) / len(ps)
                       ],
                    palette=["gold", "tab:blue", "tab:orange", "lightgray"],
                    ax=axes[1]
                )
                axes[1].set_xticks([0, 1, 2, 3], ["Positive &\nsignificant",
                                                  "Positive &\nnon-significant",
                                                  "Negative &\nsignificant",
                                                  "Negative &\nnon-significant"],
                                   fontsize=14)
                # axes[1].set_xlabel("p-value", fontsize=15)
                axes[1].set_ylabel("%", fontsize=15)
                axes[1].set_title("B) Corr. significance", weight="bold", fontsize=15)
                axes[1].legend()

                # for ax in axes:
                #     ax.set_ylabel("Count", fontsize=15)
                fig.suptitle("Correlation between Φ and self-replicators", weight="bold", fontsize=20)
                fig.tight_layout()
                plt.savefig("figures/figure_4.png")
                plt.close()


def longevity(exp="output"):
    tags = load_mat_types(binary=True)
    df = read_info(directory=exp)
    for measure in MEASURES:
        data = df[measure]
        for method in [spearmanr, kendalltau, pearsonr]:
            phis, cs = [], []
            for s, run in data.items():
                phi, c = run.flatten(), tags[s][:-1].flatten()
                if len(phi) != len(c):
                    continue
                # agg_phi.append(np.median(phi[c == 1]))
                # rep_len.append(np.sum(c))
                phis.append(phi)
                cs.append(c)

            if measure == "emergence" and method is spearmanr:
                x = np.full((len(phis), np.max([len(phi) for phi in phis])), np.nan)
                y = np.zeros((len(cs), np.max([len(c) for c in cs])))
                for i, (phi, c) in enumerate(zip(phis, cs)):
                    x[i, :len(phi)] = phi.copy()
                    y[i, :len(c)] = c.copy()
                x, y = split_by_1s_then_0s_segments(x=x, y=y)
                print(y[0])
                plt.scatter([np.nanmean(phi) for phi in x], [np.nansum(c) for c in y],
                            alpha=0.25)
                plt.xlabel("Φ", fontsize=15)
                plt.ylabel("Self-replicator length", fontsize=15)
                plt.savefig("figures/longevity.png")
                plt.close()


def combine_pvalues(exp="output", measure="emergence"):
    tags = load_mat_types(binary=True)
    df = read_info(directory=exp)
    fig, axes = plt.subplots(figsize=(16, 5), nrows=1, ncols=2)
    # for ax, measure in zip(axes, MEASURES):
    # if measure != "emergence":
    #     continue
    data = df[measure]
    p_values, effect_sizes, phi_runs, attractor_labels = [], [], [], []
    for s, run in data.items():
        phi, c = run.flatten(), tags[s][:-1].flatten()  # tags[s][:-1].flatten()
        # phi = np.diff(phi)
        if len(phi) != len(c):
            continue
        phi_att, phi_drift = phi[c != 0], phi[c == 0]
        stat, p = mannwhitneyu(phi_att, phi_drift, alternative="greater")
        p_values.append(p)
        phi_runs.append(phi)
        attractor_labels.append(c)

        all_pairs = [(a, d) for a in phi_att for d in phi_drift]
        delta = (sum(a > d for a, d in all_pairs) - sum(a < d for a, d in all_pairs)) / len(all_pairs)
        effect_sizes.append(delta)

    from scipy.stats import combine_pvalues
    combined_p = combine_pvalues(p_values, method="fisher")[1]
    print(f"{measure} combined p-value: {combined_p:.3g}")
    frac_sig = np.mean(np.array(p_values) < 0.05)
    print(f"Fraction of runs with Φ higher in attractor: {frac_sig:.2f}")
    print(f"Median Cliff’s delta = {np.median(effect_sizes):.3f}")

    anchors = np.array([0.25, 0.75])
    diffs = []
    for phi, a in zip(phi_runs, attractor_labels):
        diff = [np.mean(phi[a == 0]), np.mean(phi[a != 0])]
        diffs.append(diff)
        axes[0].plot(anchors, diff)
    diffs = np.array(diffs)
    median = np.median(diffs, axis=0)
    axes[1].plot(anchors, median)
    err = np.std(diffs, axis=0)
    axes[1].fill_between(anchors, median - err, median + err, alpha=0.25)
    for ax in axes:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-7, 3)
    axes[0].set_xticks(anchors, ["Mean Φ(drift)", "Mean Φ(self-replicating)"], fontsize=15)
    axes[0].set_ylabel("Φ per run (nats)", fontsize=15)
    axes[1].set_xticks(anchors, ["Mean Φ(drift)", "Mean Φ(self-replicating)"], fontsize=15)
    axes[1].set_ylabel("Median ± std Φ per run (nats)", fontsize=15)
    axes[0].set_title("A) Mean Φ for self-replicating and drift steps,\nper run", weight="bold", fontsize=20)
    axes[1].set_title("B) Mean Φ for self-replicating and drift steps,\nmedian±std", weight="bold", fontsize=20)
    fig.tight_layout()
    plt.savefig("figures/figure_2.png")
    plt.close()


def plot_emergence(exp="output", measure="emergence"):
    tags = load_mat_types(binary=True)
    # df = read_info(directory=exp)
    # data = df[measure]
    data = pd.read_csv("features.txt", sep=";")
    data["persistence"] = data.apply(lambda row: np.sum(tags[row["run_id"]][:-1].flatten()), axis=1)
    our_descriptors = DESCRIPTORS.copy()
    for d in ["is_flat", "max.peaks.number", "max.peaks.distance.mean", "max.peaks.distance.std", "min.peaks.number",
              "min.peaks.distance.mean", "min.peaks.distance.std", "max.peaks.val.mean", "max.peaks.val.std",
              "min.peaks.val.mean", "min.peaks.val.std", "max.min.diff.mean", "all.peaks.distance.std",
              "all.peaks.val.std"]:
        our_descriptors.remove(d)
    names = {"std": "std", "trend": "trend", "monotonicity": "monotonicity", "flatness": "flatness", "gini": "gini",
             "all.peaks.number": "number of peaks", "all.peaks.distance.mean": "mean peak distance",
             "all.peaks.val.mean": "mean peak $\Phi$"}
    for method in [spearmanr, kendalltau, pearsonr]:
        rs, ps = {desc: 0.0 for desc in our_descriptors}, {desc: 0.0 for desc in our_descriptors}
        for desc in our_descriptors:
            r, p = method(data[desc], data["persistence"])
            rs[desc] = r
            ps[desc] = p
            print(f"{desc}/{str(method)}: pvalue: {p} corr: {r}")

        if method is spearmanr:
            fig = plt.figure(figsize=(10, 6))
            plt.bar(np.arange(len(rs)), [r for r in rs.values()])
            plt.xticks(np.arange(len(rs)),
                       [names[d].replace(" ", "\n").replace("peak", "spike") for d in our_descriptors],
                       fontsize=15)
            plt.ylabel("Spearman's $\\rho$",
                       fontsize=15)
            plt.vlines(np.arange(len(rs)), [max(0, r) for r in rs.values()], [max(0, r) + 0.05 for r in rs.values()],
                       color="black")
            plt.axhline(0.0,
                        color="black", linestyle="dashed")
            plt.ylim(-0.3, 1.1)
            for i, (r, p) in enumerate(zip(rs.values(), ps.values())):
                h = max(0.0, r)
                plt.text(i - 0.24, h + 0.06, f"p={str(round(p, 4))}")
                sig = "ns"
                if 0.05 > p > 0.01:
                    sig = "*"
                elif 0.01 >= p > 0.001:
                    sig = "**"
                elif p <= 0.001:
                    sig = "***"
                plt.text(i - 0.09, h + 0.11, sig, size=15)

            plt.title("Correlation between behavior descriptors of Φ\nand persistence of self-replication",
                      weight="bold",
                      fontsize=20)
            fig.tight_layout()
            plt.savefig("figures/figure_5.png")
            plt.close()


def _cliffs_delta(x, y):
    """
    Compute Cliff's delta effect size for two arrays.
    δ = (number of x>y - number of x<y) / (n_x * n_y)
    Range [-1, 1], where:
      0.0 ≈ no effect
      0.147 ≈ small
      0.33  ≈ medium
      0.474 ≈ large
    """
    x, y = np.array(x), np.array(y)
    nx, ny = len(x), len(y)
    gt = np.sum(x[:, None] > y)
    lt = np.sum(x[:, None] < y)
    delta = (gt - lt) / (nx * ny)
    size = np.abs(delta)
    if size < 0.147:
        magnitude = "negligible"
    elif size < 0.33:
        magnitude = "small"
    elif size < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"
    return delta, magnitude


def _filter_outliers(data):
    q25, q75 = np.quantile(data, 0.25), np.quantile(data, 0.75)
    iqr_threshold = (q75 - q25) * 1.5
    return data[(q25 - iqr_threshold <= data) & (data <= q75 + iqr_threshold)]


def plot_diagnostics(data, n=1000, exp_name="figures"):
    arr = data  # np.stack(data.sample(n)["n"].to_numpy())
    sim = cosine_similarity(arr)
    plt.imshow(sim, cmap="viridis")
    plt.colorbar()
    plt.savefig(f"{exp_name}/sim_matrix.png")
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


def plot_new_molar_fractions(data):
    fig, axes = plt.subplots(figsize=(8 * data.shape[0], 5), nrows=1, ncols=data.shape[0])
    for seed, (ax, traj) in enumerate(zip(axes, data)):
        fractions = traj / traj.sum(axis=0)[np.newaxis, :]
        ax.plot(uniform_filter1d(fractions, size=10, axis=0), alpha=0.25)
        ax.set_title(str(seed), fontsize=15)
    plt.savefig("figures/fractions.png")
    plt.close()


def plot_dim_red(data):
    data = data.reshape(data.shape[0], -1)
    df = pd.read_csv("info.txt", sep=";")
    tsne = Isomap(n_components=2, metric="cosine")
    new_data = tsne.fit_transform(data)
    plt.scatter(new_data[:, 0], new_data[:, 1], c=np.median(np.array(df["synergy"].str.split('/', expand=True),
                                                                     dtype=float), axis=1))
    plt.show()


def corrected_spearmanr(a, b):
    if len(a) == 1 or len(b) == 1:
        return np.sqrt(mean_squared_error(a, b))
    if len(np.unique(a)) == 2:
        return log_loss(a.ravel(), b.ravel())  # np.mean([spearmanr(_a, _b)[0] for _a, _b in zip(a, b)])
    if np.all([i == a[0] for i in a]):
        a = a + np.random.normal(loc=0.0, scale=0.00001, size=a.shape)
    if np.all([i == b[0] for i in b]):
        b = b + np.random.normal(loc=0.0, scale=0.00001, size=b.shape)
    return spearmanr(a, b)[0]


def keep_largest_component_per_row(x, min_len=0):
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")

    out = np.zeros_like(x)

    for i, row in enumerate(x):
        padded = np.pad(row, (1, 1), constant_values=0)
        diff = np.diff(padded)

        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        if len(starts) == 0:
            continue

        lengths = ends - starts
        k = np.argmax(lengths)
        if ends[k] - starts[k] < min_len:
            continue

        out[i, starts[k]:ends[k]] = 1

    return out


def split_by_1s_then_0s_segments(x: np.ndarray, y: np.ndarray):
    """
    x, y: shape (n, m)
      - x continuous
      - y in {0,1}

    Returns
    -------
    segments : list of dict
        Each dict has:
          - run: row index
          - start, end: slice [start:end) covering 1s segment + following 0s segment
          - x_seg, y_seg: the sliced arrays (1D, length L)
          - y_event_seg: 1D, length L, with a single 1 at the first 0 in the pair
    y_event : np.ndarray, shape (n, m)
        Global transformed label: 1 at the first 0 after each 1s segment (when followed by 0s), else 0.
    """
    y_bin = (y.astype(np.int8) != 0).astype(np.int8)

    n, m = y_bin.shape
    segments = []
    y_event = np.zeros((n, m), dtype=np.int8)
    x_segs, y_events = [], []

    for i in range(n):
        yi = y_bin[i]
        xi = x[i]

        t = 0
        while t < m:
            # Find start of a 1s segment
            while t < m and yi[t] == 0:
                t += 1
            if t >= m:
                break
            ones_start = t

            # Consume 1s segment
            while t < m and yi[t] == 1:
                t += 1
            ones_end = t  # first index after the 1s segment

            # Must be followed by a 0s segment to qualify
            if ones_end >= m or yi[ones_end] != 0:
                continue

            zeros_start = ones_end  # first 0 after the 1s segment

            # Consume 0s segment
            while t < m and yi[t] == 0:
                t += 1
            zeros_end = t  # first index after the 0s segment

            # Slice covers both segments: [ones_start : zeros_end)
            start, end = ones_start, zeros_end
            x_seg = xi[start:end]
            y_seg = yi[start:end]

            # Event target: 1 at the first 0 in this pair (zeros_start), else 0
            y_event[i, zeros_start] = 1
            y_event_seg = np.zeros(end - start, dtype=np.int8)
            y_event_seg[zeros_start - start] = 1

            if np.any(y_event_seg):
                x_segs.append(x_seg)
                y_events.append(y_seg)

    x = np.full((len(x_segs), np.max([len(arr) for arr in x_segs])), np.nan)
    y = np.zeros((len(y_events), np.max([len(arr) for arr in y_events])))
    for i, (_x, _y) in enumerate(zip(x_segs, y_events)):
        x[i, :len(_x)] = _x.copy()
        y[i, :len(_y)] = _y.copy()
    return x, y.astype(int)


def train_model(exp="output", measure="emergence", ratio=1.0):
    tags = load_mat_types(binary=True)
    df = read_info(directory=exp)
    # x = pd.read_csv("features.txt", sep=";")[DESCRIPTORS]
    data = df[measure]
    x = np.full((len(data), np.max([len(arr) for arr in data.values()])), np.nan)
    y = np.zeros((len(tags), np.max([len(arr) - 1 for arr in tags.values()])))
    for i, arr in tags.items():
        x[i, :len(arr) - 1] = data[i].copy()
        y[i, :len(tags[i][:-1])] = tags[i].flatten()[:-1].copy()
    # x, y = split_by_1s_then_0s_segments(x=x, y=y)
    x = np.nan_to_num(x, nan=0.0, posinf=np.median(x[np.isfinite(x)]), neginf=np.median(x[np.isfinite(x)]))
    x = x[:, :x.shape[1] // 2]
    y = y[:, y.shape[1] // 2:]
    # x = x.reshape(-1, 1)
    # y = y.reshape(-1, 1)
    # y = np.sum(y, axis=1)
    # y = uniform_filter1d(y, size=50, axis=1)
    # forests = []
    # for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    x = StandardScaler().fit_transform(x)
    # y = StandardScaler().fit_transform(y)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    pipe = make_pipeline(
        # StandardScaler(),
        # DummyRegressor()
        MultiOutputClassifier(DummyClassifier(strategy="most_frequent"))
    )
    dummy_scores = cross_val_score(pipe, x[:, :int(x.shape[1] * ratio)], y, cv=cv,  # scoring="r2")
                                   scoring=make_scorer(lambda a, b: corrected_spearmanr(a, b), greater_is_better=True))
    # pipe = make_pipeline(
    # StandardScaler(),
    # LinearRegression()
    #     MultiOutputClassifier(RandomForestClassifier())
    # )
    # lin_scores = cross_val_score(pipe, x[:, :int(x.shape[1] * ratio)], y, cv=cv,  # scoring="r2")
    # scoring=make_scorer(lambda a, b: corrected_spearmanr(a, b), greater_is_better=True))

    pipe = make_pipeline(
        # StandardScaler(),
        MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=(64, 64),
                                            activation="relu"))
    )
    scores = cross_val_score(pipe, x[:, :int(x.shape[1] * ratio)], y, cv=cv,  # scoring="r2")
                             scoring=make_scorer(lambda a, b: corrected_spearmanr(a, b), greater_is_better=True))

    # pipe = make_pipeline(
    # StandardScaler(),
    #     MultiOutputClassifier(DecisionTreeClassifier())
    # )
    # scores = cross_val_score(pipe, x[:, :int(x.shape[1] * ratio)], y, cv=cv,  # scoring="r2")
    # scoring=make_scorer(lambda a, b: corrected_spearmanr(a, b), greater_is_better=True))
    # forests.append(scores)
    # print(ratio)
    # print(f"dummy: {dummy_scores.mean()} ± {dummy_scores.std()}")
    # print(f"linear regression: {lin_scores.mean()} ± {lin_scores.std()}")
    # print(f"mlp: {mlp_scores.mean()} ± {mlp_scores.std()}")
    # print(f"random forest: {scores.mean()} ± {scores.std()}")

    # plt.boxplot(forests)
    # plt.plot(np.arange(1, len(forests) + 1), [np.nanmedian(forest) for forest in forests], color="red")
    # plt.hlines(0.0, 0.5, len(forests) + 0.5, linestyles="dashed", color="black")
    # plt.xlabel("how early we predict", fontsize=15)
    # plt.ylabel("prediction performance", fontsize=15)
    # plt.xticks(np.arange(1, 10), ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%"])
    # plt.savefig("figures/early.png")
    # plt.close()
    # exit()
    print(mannwhitneyu(scores, dummy_scores, alternative="greater")),
    # mannwhitneyu(scores, lin_scores, alternative="greater"),
    # mannwhitneyu(scores, mlp_scores, alternative="greater"))
    plt.boxplot([scores, dummy_scores])  # mlp_scores, lin_scores, dummy_scores])
    plt.xticks([1, 2, 3, 4], ["Decision Tree", "MLP", "Linear Regression", "Dummy"])
    # plt.ylim(min([min(scores), min(lin_scores), min(dummy_scores), min(mlp_scores)]) - 0.05, 1.0)
    plt.ylabel("Prediction performance\n(Spearman's $\\rho$)", fontsize=15)
    plt.title("$\\Phi$ ~ self-replicator persistence", fontsize=15)
    plt.tight_layout()
    plt.savefig("figures/predict.png")
    plt.close()
    plt.figure(figsize=(16, 8))

    # clf = DecisionTreeRegressor().fit(x, y)
    # plot_tree(
    #     clf,  # your trained DecisionTreeClassifier
    # feature_names=feature_names,  # list of strings (optional but recommended)
    # class_names=class_names,  # list of strings (optional)
    #     filled=True,
    #     rounded=True,
    #     impurity=True
    # )
    # plt.show()

    # attention = make_pipeline(
    #     StandardScaler(),
    #     AttentionClassifier(verbose=True)
    # )
    # att_scores = cross_val_score(attention, y, x, cv=5)  # , scoring="r2")
    # print(f"Attention: {att_scores.mean()} ± {att_scores.std()}")


def plot_loss(directory="learning"):
    data = load_data(directory=directory)
    for label in ["train", "test"]:
        median = data.groupby(["i"])[".".join([label, "score"])].median()
        plt.plot(median, label=label)
        err = data.groupby(["i"])[".".join([label, "score"])].std()
        plt.fill_between(np.arange(len(median)), median - err, median + err, alpha=0.25)
    plt.xlabel("epoch", fontsize=15)
    plt.ylabel("MSE", fontsize=15)
    plt.legend()
    plt.savefig("figures/phi2c_mlp.png")
    plt.close()


def plot_learning(directory="learning"):
    data = load_data(directory=directory)
    data = data[data["method"] != "scramble"]
    data["method"] = data.apply(lambda row: {"info": "Causal\nemergence", "diff": "change\nin comp.", "fluxes": "fluxes", "history": "compositions"}[row["method"]], axis=1)
    fig, axes = plt.subplots(figsize=(20, 5 * len(data["p"].unique())), nrows=len(data["p"].unique()), ncols=2)
    for row, ((p,), d) in enumerate(data.groupby(["p"])):
        methods = []
        boxes = []
        for (method,), traj in d.groupby(["method"]):
            box = traj[traj["i"] == 99]["test.score"]  # [inner_traj["test.score"].min() for _, inner_traj in traj.groupby(["seed"])]
            boxes.append(box)
            methods.append(method)
            median = traj.groupby(traj.i)["test.score"].median()
            axes[row][0].plot(median, label=method)
            err = traj.groupby(traj.i)["test.score"].std()
            axes[row][0].fill_between(np.arange(len(median)), median - err, median + err, alpha=0.25)
        axes[row][0].hlines(traj[traj["i"] == traj["i"].max()]["dummy.score"], xmin=0, xmax=traj["i"].max(),
                            label="baseline",
                            color="black", linestyles="dashed")
        boxes.append(traj[traj["i"] == traj["i"].max()]["dummy.score"])
        methods.append("baseline")

        for m1, b1 in zip(methods, boxes):
            for m2, b2 in zip(methods, boxes):
                if m1 == m2 or m1 != "info":
                    continue
                print(f"{p} - {m1} vs. {m2}: {mannwhitneyu(b1, b2, alternative='less')}")

        axes[row][1].boxplot(boxes)
        for ax in axes[row]:
            ax.set_ylabel("log loss (the lower, the better)", fontsize=15)
            ax.set_title(str(p), fontsize=10)
        axes[row][1].set_xticks(np.arange(1, len(methods) + 1), methods, fontsize=15)
        axes[row][0].legend()
    plt.savefig("figures/learning.png")
    plt.close()


if __name__ == "__main__":
    # d = load_all_mat_data(attribute="fluxes")
    # plot_new_molar_fractions(data=d[:10, :99, :])
    # exit()
    # d = load_data()
    # d = d[d["i"] < 500]
    # plot_info()
    # plot_samples()
    # plot_info_tags()
    # plot_neighborhood_histogram()
    # granger_causality()
    # granger_and_descriptors()
    # plot_correlations()
    # plot_mixed_effects()
    # plot_corr()
    # longevity()
    # plot_emergence()
    # combine_pvalues()
    # train_model()
    plot_learning()
    # plot_pca_evr()
    # d = d.transpose(0, 2, 1).reshape(-1, d.shape[1])
    # idx = np.random.choice(d.shape[0], size=1000, replace=False)
    # sampled = d[idx, :]  # shape: (n_samples, n)
    # plot_diagnostics(data=d.reshape(d.shape[1], -1))
    # plot_dim_red(data=d)
    # n_seeds = 10
    # d = d[d["seed"] < n_seeds]
    # d = d[d["is_daughter"]]
    # plot_molar_fractions(data=d)
    # plot_asymptotic_sim(data=d)
    # for i in range(n_seeds):
    #     carpet_plot((d, i))
    # plot_loss()
