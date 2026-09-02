import os
from functools import reduce

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mtick
import seaborn as sns
from scipy.io import loadmat, savemat
from scipy.ndimage import uniform_filter1d, minimum_filter1d
from scipy.signal import find_peaks
from scipy.stats import linregress, spearmanr, kendalltau, pearsonr, zscore, wilcoxon, ttest_1samp, mannwhitneyu, norm, \
    ttest_ind, ttest_rel, entropy, t
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
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.formula.api as smf
import statsmodels.api as sm

from utils import MEASURES, set_seed, DESCRIPTORS, get_info_array, NMAX

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
            if "learning" in directory:
                curr_d["method"] = file.split(".")[1]
                curr_d["p"] = float(file.split(".")[0].replace(',', '.'))
                curr_d["seed"] = int(file.split(".")[-2])
            elif "causality" in directory:
                curr_d["seed"] = int(file.split(".")[0])
                curr_d["p"] = float(file.split(".")[1].replace(",", "."))
                curr_d["exp"] = file.split(".")[2]
            elif "optim" in directory:
                curr_d["trace"] = int(file.split(".")[0])
                curr_d["seed"] = int(file.split(".")[1])
                curr_d["exp"] = file.split(".")[2]
            else:
                curr_d["seed"] = int(file.split(".")[-2])
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
    data = {}
    for num, file in enumerate(os.listdir(path)):
        if not file.endswith("mat"):
            continue
        curr_d = loadmat(os.path.join(path, file))
        trace = curr_d["o"][attribute][0, 0]
        data[int(file.split("_")[-1].split(".")[0])] = trace
    return data


def load_all_mat_data_branching(path=os.path.join(os.getcwd(), "GARD-model", "Matlab", "SourceCode",
                                                  "Doron_Lancet_GARD_Next_generation", "GARD_v10", "new_causality"),
                                reduced=True):
    data = []
    for num, file in enumerate(os.listdir(path)):
        if not file.endswith("mat"):
            continue
        branch_id = int(file.split("_")[5].split(".")[0])
        if reduced and branch_id <= 30:
            continue
        curr_d = loadmat(os.path.join(path, file))
        trace = curr_d["o"]["history"][0, 0]
        tags = curr_d["o"]["tags"][0, 0]
        data.append({"mother.id": int(file.split("_")[3]),
                     "branch.id": int(file.split("_")[5].split(".")[0]),
                     "trace": trace,
                     "tags": tags})
    data = pd.DataFrame(data)
    return data


def load_all_mat_data_causality(directory="causality"):
    data = []
    for num, file in enumerate(os.listdir(directory)):
        if not file.endswith("txt"):
            continue
        try:
            trace = loadmat(os.path.join(directory, "traces", file.replace("txt", "mat")))["history"]
            tags = loadmat(os.path.join(directory, "tags", file.replace("txt", "mat")))["tags"].ravel()
        except FileNotFoundError:
            continue
        data.append({"seed": int(file.split(".")[0]),
                     "p": float(file.split(".")[1].replace(",", ".")),
                     "trace": trace,
                     "tags": tags})
    data = pd.DataFrame(data)
    return data


def load_mat_types(path=os.path.join(os.getcwd(), "GARD-model", "Matlab", "SourceCode",
                                     "Doron_Lancet_GARD_Next_generation", "GARD_v10", "data"),
                   binary=False):
    data = {}
    for num, file in enumerate(os.listdir(path)):
        if not file.endswith("mat"):
            continue
        curr_d = loadmat(os.path.join(path, file))
        if "causality" in path:
            tags = curr_d["tags"].ravel()
        else:
            tags = curr_d["o"]["tags"][0, 0]
        if binary:
            tags = np.where(tags != 0, 1, 0)
        data[num] = tags
    return data


def load_train_tags(path):
    data = []
    for file_name in os.listdir(path):
        d = loadmat(os.path.join(path, file_name))
        trace = loadmat(os.path.join(path, file_name).replace("test", "traces"))["history"]
        tags, ttrain, ttest = d["tags"].ravel().astype(int), d["ttrain"][0][0], d["ttest"][0][0]
        seed_sim = int(file_name.split(".")[0])
        long_d = loadmat(
            f"GARD-model/Matlab/SourceCode/Doron_Lancet_GARD_Next_generation/GARD_v10/data_long/GARD_run_seed_{seed_sim:03d}.mat")[
            "o"]
        long_trace = long_d["history"][0, 0]
        long_tags = long_d["tags"][0, 0].astype(int)
        long_tags = np.where(long_tags >= 1, 1, 0)
        gen_idx = np.where(np.sum(long_trace, axis=0) == NMAX)[0]
        third_idx = len(gen_idx) // 3
        metadata = pd.read_csv(os.path.join(path, file_name).replace("test/", "").replace("mat", "txt"), sep=";")
        try:
            data.append({"seed.sim": seed_sim,
                         "seed.evo": int(file_name.split(".")[1]),
                         "exp": file_name.split(".")[2],
                         "relax.tags": tags[:ttrain],
                         "train.tags": tags[ttrain: ttest],
                         "test.tags": tags[ttest:],
                         "trace": trace,
                         "long.trace": long_trace[:, third_idx: third_idx * 2],
                         "long.relax.tags": long_tags[:third_idx].ravel(),
                         "long.train.tags": long_tags[third_idx: third_idx * 2].ravel(),
                         "long.test.tags": long_tags[third_idx * 2:].ravel(),
                         "is_rogue": third_idx != 100,
                         "fitness": float(metadata.iloc[-1]["best.fitness"]),
                         "fitness.delta": float(metadata.iloc[-1]["best.fitness"] - metadata.iloc[0]["best.fitness"])
                         })
        except IndexError:
            continue
    return pd.DataFrame(data)


def load_causality_tags(path, by_time=False):
    data = []
    for file_name in os.listdir(path):
        if not file_name.endswith("txt"):
            continue
        seed = int(file_name.split(".")[0])
        metadata = pd.read_csv(os.path.join(path, file_name), sep=";")
        tr = loadmat(os.path.join(path, "traces", file_name.replace("txt", "mat")))["history"].T
        try:
            tags = loadmat(os.path.join(path, "tags", file_name.replace("txt", "mat")))["tags"]["tags"][0, 0].ravel()
        except IndexError:
            tags = loadmat(os.path.join(path, "tags", file_name.replace("txt", "mat")))["tags"].ravel()
        exp = file_name.split(".")[2]
        if exp == "base":
            exp = "control"
        if not by_time:
            data.append({"seed": seed,
                         "exp": exp,
                         "trace": tr,
                         "tags": tags.astype(int),
                         "phi": float(metadata.iloc[-1]["phi"]),
                         "phi.delta": float(metadata.iloc[-1]["phi"] - metadata.iloc[0]["phi"])
                         })
        else:
            metadata["step"] = metadata["step"] + 3
            metadata["prev_step"] = metadata["step"].shift(1).fillna(0).astype(np.int32)
            for (gen,), traj in metadata.groupby(["gen"]):
                s1, s2 = traj["prev_step"].item(), traj["step"].item()
                data.append({"seed": seed,
                             "exp": exp,
                             "i": gen,
                             "step": traj["step"].item(),
                             "trace": tr[:, s1: s2],
                             "tags": tags.astype(int)[s1: s2],
                             "phi": float(traj["phi"].item())
                             })
    return pd.DataFrame(data)


def load_lifeless(path):
    data = []
    for file_name in os.listdir(os.path.join(path, "traces")):
        if not file_name.endswith("mat"):
            continue
        seed = int(file_name.split(".")[0])
        tr = loadmat(os.path.join(path, "traces", file_name))["history"].T
        try:
            tags = loadmat(os.path.join(path, "tags", file_name))["tags"].ravel()
        except FileNotFoundError:
            continue
        exp = file_name.split(".")[-2]
        is_base = exp == "base"
        if is_base:
            exp = "control"
        data.append({"seed": seed,
                     "exp": exp,
                     "kf": file_name.split(".")[1 if is_base else 2].replace(",", "."),
                     "kb": file_name.split(".")[2 if is_base else 3].replace(",", "."),
                     # "A": file_name.split(".")[3],
                     # "sigma": file_name.split(".")[4],
                     "Nmax": file_name.split(".")[5 if is_base else 4],
                     "trace": tr,
                     "tags": tags.astype(int)
                     })
    return pd.DataFrame(data)


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


def read_info_branching(directory="output_branching"):
    info = []
    for measure in MEASURES:
        if measure == "emergence":
            continue
        root = os.path.join(directory, measure.lower())
        for file in os.listdir(os.path.join(directory, measure.lower())):
            d = np.load(os.path.join(root, file))
            info.append({"mother.id": int(file.split(".")[0]),
                         "branch.id": int(file.split(".")[1]),
                         "measure": measure,
                         "info": d})
    # data = {}
    # for i in info["synergy"].keys():  # arr1, arr2 in zip(info["synergy"], info["causation"]):
    #     arr1 = info["synergy"][i]
    #     arr2 = info["causation"][i]
    # min_length = min(len(arr1[~np.isnan(arr1)]), len(arr2[~np.isnan(arr2)]))
    #     emergence = np.nansum([arr1, arr2], axis=0).flatten()
    # emergence[min_length:] = np.nan
    # data.append(emergence)
    #     data[i] = emergence
    # info["emergence"] = data  # np.array(data)
    # print(info["emergence"].shape, info["synergy"].shape, info["causation"].shape)
    info = pd.DataFrame(info)
    for (midx,), traj in info.groupby(["mother.id"]):
        for (bidx,), d in traj.groupby(["branch.id"]):
            try:
                emergence = np.nansum([d[d["measure"] == "synergy"]["info"].item(),
                                       d[d["measure"] == "causation"]["info"].item()],
                                      axis=0).flatten()
                info = pd.concat([info, pd.DataFrame([{"mother.id": midx,
                                                       "branch.id": bidx,
                                                       "measure": "emergence",
                                                       "info": emergence}])],
                                 ignore_index=True)
            except ValueError:
                continue
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


def other_metrics(exp="output", measure="emergence", method=spearmanr):
    info = read_info(directory=exp)[measure]
    network = pd.read_csv("networks.txt", sep=";")
    for f in network.columns:
        x = [np.nanmedian(info[run]) for (run,), _ in network.groupby(["seed"])]
        y = network[f]
        print(f"{f}: {method(x, y)}")
    print("#########")
    dynamics = pd.read_csv("dynamics.txt", sep=";")
    for col in dynamics.columns:
        dynamics[col] = np.nan_to_num(dynamics[col],
                                      nan=np.nanmedian(dynamics[col]),
                                      posinf=np.nanmax(dynamics[col]),
                                      neginf=np.nanmin(dynamics[col]))
    for f in dynamics.columns:
        x = [np.nanmedian(info[run]) for (run,), _ in dynamics.groupby(["seed"])]
        y = dynamics[f]
        print(f"{f}: {method(x, y)}")


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
    fig, axes = plt.subplots(figsize=(16, 10), nrows=2, ncols=2)
    runs = np.random.randint(0, len(df[measure]), size=n)
    data = get_info_array(df=df, measure=measure)
    median = np.nanmedian(data, axis=0)
    median = uniform_filter1d(median, size=1)
    axes[0][0].plot(median)
    x = np.arange(len(median))
    beta_1, beta_0, rvalue, pvalue, _ = linregress(x, median)
    y_hat = beta_0 + (beta_1 * x)
    axes[0][0].plot(x, y_hat, color="red", label=f"p={round(pvalue, 4)} > 0.05")
    axes[0][0].legend()
    err = np.nanstd(data, axis=0)
    axes[0][0].fill_between(x, median - err, median + err, alpha=0.25)

    k = 0
    for i in range(2):
        for j in range(2):
            if i == 0 and j == 0:
                continue
            d = df[measure][runs[k]]
            axes[i][j].plot(d)
            k += 1

    for i in range(1, 2):
        for j in range(2):
            axes[i][j].set_xlabel("molecular step", fontsize=15)
            axes[i][j].set_ylabel("$\\Phi^r$ (nats)", fontsize=15)
    axes[0][0].set_title("A) Median±std of $\\Phi^r$ over 100 runs", weight="bold", fontsize=20)
    # for ax, title in zip(axes[1:], ["B", "C", "D"]):
    axes[0][1].set_title(f"B) Sample run", weight="bold", fontsize=20)
    axes[1][0].set_title(f"C) Sample run", weight="bold", fontsize=20)
    axes[1][1].set_title(f"D) Sample run", weight="bold", fontsize=20)

    fig.tight_layout()
    plt.savefig("figures/figure_1.png")
    plt.close()


def check_noise(exp="output", measure="emergence"):
    df = read_info(directory=exp)[measure]
    for seed, run in df.items():
        run = StandardScaler().fit_transform(run.reshape(-1, 1))
        plt.plot(run)
        plt.show()
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
            mtick.FuncFormatter(lambda y, _: f"{y * 100:g}")
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
                fig.suptitle("Correlation between $\\Phi^r$ and self-replicators", weight="bold", fontsize=20)
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
    print(f"Fraction of runs with $\\Phi^r$ higher in attractor: {frac_sig:.2f}")
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
    axes[0].set_xticks(anchors, ["Mean $\\Phi^r$(drift)", "Mean $\\Phi^r$(self-replicating)"], fontsize=15)
    axes[0].set_ylabel("$\\Phi^r$ per run (nats)", fontsize=15)
    axes[1].set_xticks(anchors, ["Mean $\\Phi^r$(drift)", "Mean $\\Phi^r$(self-replicating)"], fontsize=15)
    axes[1].set_ylabel("Median ± std $\\Phi^r$ per run (nats)", fontsize=15)
    axes[0].set_title("A) Mean $\\Phi^r$ for self-replicating and drift steps,\nper run", weight="bold", fontsize=20)
    axes[1].set_title("B) Mean $\\Phi^r$ for self-replicating and drift steps,\nmedian±std", weight="bold", fontsize=20)
    fig.tight_layout()
    plt.savefig("figures/figure_2.png")
    plt.close()


def plot_emergence(exp="output", measure="emergence", var="probability"):
    tags = load_mat_types(binary=True)
    # df = read_info(directory=exp)
    # data = df[measure]
    data = pd.read_csv("new_features.txt", sep=";")
    data["persistence"] = data.apply(lambda row: np.sum(tags[row["run_id"]][:-1].flatten()), axis=1)
    data["probability"] = data.apply(lambda row: np.mean(tags[row["run_id"]][:-1].flatten()), axis=1)
    our_descriptors = DESCRIPTORS.copy()
    for d in ["is_flat", "max.peaks.number", "max.peaks.distance.mean", "max.peaks.distance.std", "min.peaks.number",
              "min.peaks.distance.mean", "min.peaks.distance.std", "max.peaks.val.mean", "max.peaks.val.std",
              "min.peaks.val.mean", "min.peaks.val.std", "max.min.diff.mean", "all.peaks.distance.std",
              "all.peaks.val.std", "all.peaks.time.std", "min.peaks.time.std", "max.peaks.time.std",
              "max.peaks.time.mean", "min.peaks.time.mean"]:
        try:
            our_descriptors.remove(d)
        except:
            print(d)
            raise
    names = {"std": "std", "trend": "trend", "monotonicity": "monotonicity", "flatness": "flatness", "gini": "gini",
             "all.peaks.number": "number of peaks", "all.peaks.distance.mean": "mean peak distance",
             "all.peaks.val.mean": "mean peak $\Phi$", "all.peaks.time.mean": "mean peak time"}
    for method in [spearmanr, kendalltau, pearsonr]:
        rs, ps = {desc: 0.0 for desc in our_descriptors}, {desc: 0.0 for desc in our_descriptors}
        for desc in our_descriptors:
            r, p = method(data[desc], data[var])
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
                plt.text(i - 0.24, h + 0.06, f"$\\rho$={round(r, 4)}, p={str(round(p, 4))}")
                sig = "ns"
                if 0.05 > p > 0.01:
                    sig = "*"
                elif 0.01 >= p > 0.001:
                    sig = "**"
                elif p <= 0.001:
                    sig = "***"
                plt.text(i - 0.09, h + 0.11, sig, size=15)

            plt.title(f"Correlation between behavior descriptors of Φ\nand {var} of self-replication",
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


def filter_noise(row, min_len=10):
    comp = np.zeros_like(row)
    padded = np.pad(row, (1, 1), constant_values=0)
    diff = np.diff(padded)

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    if len(starts) == 0:
        return comp
    lengths = ends - starts

    for k, length in enumerate(lengths):
        if length >= min_len:
            comp[starts[k]:ends[k]] = 1
    return comp


def keep_largest_component(row, min_len=0, n=1, by_type=False):
    if by_type:
        values, counts = np.unique_counts(row)
        k = np.argmax(counts)
        comp = (row == values[k]).astype(int)
        return comp
    comp = np.zeros_like(row)
    padded = np.pad(row, (1, 1), constant_values=0)
    diff = np.diff(padded)

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    if len(starts) == 0:
        return comp
    lengths = ends - starts
    # k = np.argmax(lengths)
    for k in np.argsort(lengths)[-n:]:
        if ends[k] - starts[k] < min_len:
            return comp

        comp[starts[k]:ends[k]] = 1
    return comp


def keep_largest_component_per_row(x, min_len=0):
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("x must be a 2D array")

    out = np.zeros_like(x)

    for i, row in enumerate(x):
        out[i] = keep_largest_component(row=row, min_len=min_len)

    return out


def find_earliest_component(row, keep_largest=True, by_type=False):
    if np.all(row == 0):
        return 1.0
    if keep_largest:
        return np.where(keep_largest_component(row=row.astype(np.int32), by_type=by_type) == 1)[0][0] / len(row)
    return np.where(row.astype(np.int32) == 1)[0][0] / len(row)


def number_of_runs(x):
    # x = np.asarray(x).astype(int)
    # padded = np.r_[0, x, 0]
    # starts = np.where(np.diff(padded) != 1)[0]
    # ends = np.where(np.diff(padded) == -1)[0]
    # lengths = ends - starts
    # return lengths.mean() if len(lengths) else 0
    # return 1 - np.mean(x[1:] != x[:-1])
    # from zlib import compress
    # return len(compress(x.tobytes()))
    return np.corrcoef(x[:-1], x[1:])[0, 1]
    # runs = 1 + np.sum(x[:-1] != x[1:])
    # return 1 - (runs - 1)/(len(x)-1)


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
        t = tags[i].flatten()[:-1].copy()
        y[i, :len(t)] = np.where(keep_largest_component(t) == 1)[0][0] / len(t)
    # x, y = split_by_1s_then_0s_segments(x=x, y=y)
    x = np.nan_to_num(x, nan=0.0, posinf=np.median(x[np.isfinite(x)]), neginf=np.median(x[np.isfinite(x)]))
    y = y[:, 0]
    # x = x[:, :x.shape[1] // 2]
    # y = y[y.shape[0] // 2:]
    # x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    # y = np.sum(y, axis=1)
    # y = uniform_filter1d(y, size=50, axis=1)
    # forests = []
    # for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    x = StandardScaler().fit_transform(x)
    # y = StandardScaler().fit_transform(y)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    pipe = make_pipeline(
        # StandardScaler(),
        DummyRegressor()
        # MultiOutputClassifier(DummyClassifier(strategy="most_frequent"))
    )
    dummy_scores = cross_val_score(pipe, x[:, :int(x.shape[1] * ratio)], y, cv=cv,  # scoring="r2")
                                   scoring=make_scorer(lambda a, b: corrected_spearmanr(a, b), greater_is_better=True))
    pipe = make_pipeline(
        # StandardScaler(),
        LogisticRegression()
        #     MultiOutputClassifier(RandomForestClassifier())
    )
    lin_scores = cross_val_score(pipe, x[:, :int(x.shape[1] * ratio)], y, cv=cv,  # scoring="r2")
                                 scoring=make_scorer(lambda a, b: corrected_spearmanr(a, b), greater_is_better=True))

    pipe = make_pipeline(
        # StandardScaler(),
        MLPRegressor()
        # MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=(64, 64),
        #                                     activation="relu"))
    )
    mlp_scores = cross_val_score(pipe, x[:, :int(x.shape[1] * ratio)], y, cv=cv,  # scoring="r2")
                                 scoring=make_scorer(lambda a, b: corrected_spearmanr(a, b), greater_is_better=True))

    pipe = make_pipeline(
        # StandardScaler(),
        DecisionTreeRegressor()
        #     MultiOutputClassifier(DecisionTreeClassifier())
    )
    scores = cross_val_score(pipe, x[:, :int(x.shape[1] * ratio)], y, cv=cv,  # scoring="r2")
                             scoring=make_scorer(lambda a, b: corrected_spearmanr(a, b), greater_is_better=True))
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
    print(mannwhitneyu(scores, dummy_scores, alternative="greater"),
          mannwhitneyu(scores, lin_scores, alternative="greater"),
          mannwhitneyu(scores, mlp_scores, alternative="greater"))
    plt.boxplot([scores, mlp_scores, lin_scores, dummy_scores])
    plt.xticks([1, 2, 3, 4], ["Decision Tree", "MLP", "Linear Regression", "Dummy"])
    # plt.ylim(min([min(scores), min(lin_scores), min(dummy_scores), min(mlp_scores)]) - 0.05, 1.0)
    plt.ylabel("Prediction performance\n(Spearman's $\\rho$)", fontsize=15)
    plt.title("$\\Phi$ ~ self-replicator persistence", fontsize=15)
    plt.tight_layout()
    plt.savefig("figures/predict.png")
    plt.close()
    # plt.figure(figsize=(16, 8))

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


def plot_learning(directory="new_learning"):
    data = load_data(directory=directory)
    data = data[data["method"] != "scramble"]
    data["method"] = data.apply(lambda row:
                                {"info": "Causal\nemergence", "diff": "change\nin comp.", "fluxes": "fluxes",
                                 "history": "compositions"}[row["method"]], axis=1)
    fig, axes = plt.subplots(figsize=(20, 5 * len(data["p"].unique())), nrows=len(data["p"].unique()), ncols=2)
    for row, ((p,), d) in enumerate(data.groupby(["p"])):
        methods = []
        boxes = []
        for (method,), traj in d.groupby(["method"]):
            box = traj[traj["i"] == 99][
                "test.score"]  # [inner_traj["test.score"].min() for _, inner_traj in traj.groupby(["seed"])]
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
    plt.savefig("figures/new_learning.png")
    plt.close()


def plot_learning_pretty(directory="new_learning_logistic"):
    data = load_data(directory=directory)
    data = data[data["method"] != "scramble"]
    data["method"] = data.apply(lambda row: {"info": "$\\Phi^r$", "diff": "change\nin comp.",
                                             "fluxes": "fluxes", "history": "compositions"}[row["method"]], axis=1)
    data = data[data["p"] == 0.5]
    fig, axes = plt.subplots(figsize=(8, 5), nrows=1, ncols=1)
    methods = []
    boxes = []
    for (method,), traj in data.groupby(["method"]):
        box = traj[traj["i"] == 9]["test.score"]
        # box = traj.groupby(traj.seed)["test.score"].max()
        boxes.append(box)
        methods.append(method)
    boxes.append(traj[traj["i"] == traj["i"].max()]["dummy.score"])
    methods.append("baseline")

    for m1, b1 in zip(methods, boxes):
        for m2, b2 in zip(methods, boxes):
            if m1 == m2 or m1 != "$\\Phi^r$":
                continue
            print(f"{m1} vs. {m2}: {mannwhitneyu(b1, b2, alternative='greater')}")
    axes.boxplot(boxes)

    hook = 0.005
    for i, (m1, b1) in enumerate(zip(methods, boxes)):
        for j, (m2, b2) in enumerate(zip(methods, boxes)):
            if m1 == m2 or m1 != "$\\Phi^r$":
                continue
            mw = mannwhitneyu(b1, b2, alternative="greater")
            y = 0.9 + (0.05 * (j - 1))
            axes.plot([i + 1, i + 1, j + 1, j + 1], [y, y + hook, y + hook, y], lw=1.5, c='black')
            axes.text((i + 1 + j + 1) * 0.5, y - 0.001, "***" if mw.pvalue < 0.001 else "**", ha='center',
                      va='bottom', fontsize=15)

    axes.set_ylabel("Accuracy (self-replication vs. not)", fontsize=15)
    axes.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    axes.set_title("A model trained on $\\Phi^r$ predicted\nappearance of self-replicators better than all baselines",
                   weight="bold", fontsize=20)
    axes.set_xticks(np.arange(1, len(methods) + 1), methods, fontsize=15)
    # axes.set_ylim(-5, 0)  # 500, 15000)
    axes.legend()
    fig.tight_layout()
    plt.savefig("figures/new_learning_logistic.png")
    plt.close()


def plot_branching_analysis(measure="emergence"):
    data = load_all_mat_data_branching()
    info = read_info("output_mothers")[measure]
    info = pd.DataFrame({"mother.id": list(info.keys()), "branch.id": 0, "info": list(info.values())})
    # info = info[info["measure"] == measure]
    x, y, z = [], [], []
    for (idx,), traj in data.groupby(["mother.id"]):
        x.extend(  # np.nanmean(
            [np.nanmedian(d["info"].item()) for _, d in info[info["mother.id"] == idx].groupby(["branch.id"])]
        )  #)
        # y.append(np.nansum(info[idx - 1]))
        # beta_1, beta_0, rvalue, pvalue, _ = linregress([np.nanmedian(info[idx]) for _, d in traj.groupby(["branch.id"])],  # [np.nansum(d["info"].item()) for _, d in info[info["mother.id"] == idx].groupby(["branch.id"])],
        #                                                [np.mean(d["tags"].item()) for _, d in traj.groupby(["branch.id"])])
        # x.append(pvalue)
        y.extend(  # np.mean(
            [np.nanmean(d["tags"].item()) for _, d in traj.groupby(["branch.id"])]
            # [np.nansum(keep_largest_component(d["tags"].item().astype(np.int32).ravel())) for _, d in traj.groupby(["branch.id"])]
            # [find_earliest_component(d["tags"].item().astype(np.int32).ravel()) for _, d in traj.groupby(["branch.id"])]
        )  # )
        z.extend([idx] * len(traj))
    x, y = np.array(x), np.array(y)

    for method in [kendalltau, pearsonr, spearmanr]:
        print(method(x, y))
    # print(np.mean(y <= 0.05))
    plt.scatter(x, y, alpha=0.25)
    # categories = pd.Categorical(z)
    # colors = plt.cm.tab10(categories.codes)
    # for cat, color in zip(categories.categories, colors):
    #     plt.plot(x[z == cat], y[z == cat], color=color, alpha=0.25)
    # plt.hist(x)
    beta_1, beta_0, rvalue, pvalue, _ = linregress(x, y)
    y_hat = beta_0 + (beta_1 * x)
    plt.plot(x, y_hat, color="red", label=f"r={round(rvalue, 3)}, p={round(pvalue, 4)}")
    plt.ylabel("self-replication persistence", fontsize=15)
    plt.xlabel("branch $\\Phi$", fontsize=15)
    plt.legend()
    plt.savefig("figures/prob_vs_phi.png")
    plt.close()

    # all_data = pd.merge(data, info, on=["mother.id", "branch.id"])
    # all_data["phi"] = all_data.apply(lambda row: np.nanmedian(row["info"]), axis=1)
    # all_data["phi"] = all_data.apply(lambda row: np.nanmedian(info[row["mother.id"]]), axis=1)
    # all_data["p"] = all_data.apply(lambda row: np.nanmean(keep_largest_component(row["tags"].item().astype(np.int32).ravel())),
    #                                axis=1)
    # all_data["phi"] = (all_data["phi"] - all_data["phi"].mean()) / all_data["phi"].std()
    # all_data["p"] = (all_data["p"] - all_data["p"].mean()) / all_data["p"].std()
    x = StandardScaler().fit_transform(x.reshape(-1, 1)).ravel()
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
    df = pd.DataFrame({"phi": x, "p": y, "mother.id": z})
    independent, dependent = "phi", "p"
    model = smf.mixedlm(f"{independent} ~ {dependent}", data=df, groups=df["mother.id"],
                        # re_formula=f"~{dependent}"
                        ).fit()
    print(f"{dependent}: slope: {model.params[dependent]} p-value: {model.pvalues[dependent]}")
    independent, dependent = "p", "phi"
    model = smf.mixedlm(f"{independent} ~ {dependent}", data=df, groups=df["mother.id"],
                        # re_formula=f"~{dependent}"
                        ).fit()
    print(f"{dependent}: slope: {model.params[dependent]} p-value: {model.pvalues[dependent]}")


def plot_branching_analysis_bis(measure="emergence"):
    data = load_all_mat_data_branching()
    old_data = load_all_mat_data()
    info = read_info_branching("output_branching")
    info = info[info["measure"] == measure]
    data = pd.merge(data, info, on=["mother.id", "branch.id"], how="inner")
    data["phi"] = data.apply(lambda row: np.nanmedian(row["info"]), axis=1)
    for idx, row in data.iterrows():
        if old_data[row["mother.id"]].shape[1] // 2 + (len(row["tags"]) // 10) >= len(row["tags"]):
            print("BOOM")
    data["prob"] = data.apply(lambda row: np.nansum(
        row["tags"].ravel()[old_data[row["mother.id"]].shape[1] // 2 + (len(row["tags"]) // 10):]), axis=1)
    # data["prob"] = data.apply(lambda row: np.nanmean(keep_largest_component(row["tags"].astype(np.int32).ravel())),
    #                           axis=1)
    data = data[data["tags"].apply(lambda x: len(x) > 0)]
    fig, axes = plt.subplots(figsize=(24, 5), nrows=1, ncols=3, sharey=True)
    probs = []
    # data = data[data["mother.id"] <= 30]
    for ax, rule in zip(axes, [lambda d, m: d,  # .loc[d["branch.id"] == m],
                               lambda d, m: d.loc[d["phi"].idxmax()],
                               lambda d, m: d.loc[d["phi"].idxmin()]]):
        x, y, z = [], [], []
        for (idx,), traj in data.groupby(["mother.id"]):
            chosen = rule(traj, idx)
            if not len(chosen):
                continue
            x.append(chosen["phi"].mean())
            y.append(chosen["prob"].mean())
            # [np.nansum(keep_largest_component(d["tags"].item().astype(np.int32).ravel())) for _, d in traj.groupby(["branch.id"])]
            # [find_earliest_component(d["tags"].item().astype(np.int32).ravel()) for _, d in traj.groupby(["branch.id"])]
            # ))
            z.append(idx)
        x, y = np.array(x), np.array(y)
        probs.append(y)

        df = pd.DataFrame({"phi": x, "p": y, "mother.id": z})
        df.dropna(how="any", inplace=True, axis=0)
        independent, dependent = "phi", "p"
        model = smf.mixedlm(f"{independent} ~ {dependent}", data=df, groups=df["mother.id"],
                            # re_formula=f"~{dependent}"
                            ).fit()
        print(f"{dependent}: slope: {model.params[dependent]} p-value: {model.pvalues[dependent]}")
        independent, dependent = "p", "phi"
        model = smf.mixedlm(f"{independent} ~ {dependent}", data=df, groups=df["mother.id"],
                            # re_formula=f"~{dependent}"
                            ).fit()
        print(f"{dependent}: slope: {model.params[dependent]} p-value: {model.pvalues[dependent]}")

        for method in [kendalltau, pearsonr, spearmanr]:
            print(method(x, y))
        ax.scatter(x, y)
        beta_1, beta_0, rvalue, pvalue, _ = linregress(x, y)
        y_hat = beta_0 + (beta_1 * x)
        ax.plot(x, y_hat, color="red", label=f"r={round(rvalue, 3)}, p={round(pvalue, 4)}")

        ax.legend()

    plt.show()
    plt.close()

    for i, p1 in enumerate(probs):
        for j, p2 in enumerate(probs):
            if i == j:
                continue
            print(i, j,
                  wilcoxon(p1, p2),
                  mannwhitneyu(p1, p2, alternative="greater"),
                  _cliffs_delta(p1, p2))

    fig, axes = plt.subplots(figsize=(20, 5), nrows=1, ncols=2)
    axes[0].violinplot(probs)
    axes[1].boxplot(probs)
    for ax in axes:
        ax.set_xticks([1, 2, 3], ["base", "max", "min"])

    plt.show()


def fit_poly(x, y, degree):
    X = np.column_stack([x ** i for i in range(1, degree + 1)])
    X = sm.add_constant(X)
    return sm.OLS(y, X).fit()


def plot_causality_analysis(directory="new_causality_max", method=spearmanr):
    metadata = load_data(directory=directory)
    data = load_all_mat_data_causality(directory=directory)
    metadata.replace([np.inf, -np.inf], np.nan, inplace=True)
    all_data = pd.merge(metadata, data, on=["seed"], how="inner")
    all_data["prev_step"] = all_data["step"].shift(1).fillna(0).astype(np.int32)
    all_data = all_data[all_data["n_sum"] != 0]
    all_data["prob"] = all_data.apply(lambda row: np.nansum(row["tags"][row["prev_step"]: row["step"]]),
                                      axis=1)
    all_data.dropna(axis=0, how="any", inplace=True)
    # all_data = all_data[all_data["gen"] <= 50]
    # all_data["phi"] = (all_data["phi"] - all_data["phi"].mean()) / all_data["phi"].std()
    # all_data["prob"] = (all_data["prob"] - all_data["prob"].mean()) / all_data["prob"].std()
    # fig, axes = plt.subplots(figsize=(20, 5), nrows=1, ncols=2)
    # for i, (label, col) in enumerate(zip(["$\\Phi$", "self-replication"], ["phi", "prob"])):
    #     median = all_data.groupby(["gen"])[col].median()
    #     axes[i].plot(np.arange(len(median)), median, label=label)
    #     err = all_data.groupby(["gen"])[col].std()
    #     axes[i].fill_between(np.arange(len(median)), median - err, median + err, alpha=0.25)
    #     axes[i].legend()
    # plt.show()
    # plt.close()

    x, y = [], []
    for (seed,), traj in all_data.groupby(["seed"]):
        # y.append(
        #     np.nanmean([np.nansum(d["info"].item()) for _, d in info[info["mother.id"] == idx].groupby(["branch.id"])]))
        a, b = traj["phi"].values, traj["tags"].values
        # b = uniform_filter1d(b, size=len(b) // len(a))
        # indices = np.linspace(0, len(b) - 1, len(a)).astype(int)
        # b = b[indices]
        # a = StandardScaler().fit_transform(a.reshape(-1, 1)).ravel()
        # b = StandardScaler().fit_transform(b.reshape(-1, 1)).ravel()
        y.append(np.nanmean(a))
        x.append(np.nanmean(b[0]))
        continue
        beta_1, beta_0, rvalue, pvalue, _ = linregress(a, b)
        try:
            res = method(a, b)
        except ValueError:
            continue
        # x.append(res.pvalue)
        # y.append(res.statistic)
        x.append(pvalue)
        y.append(rvalue)
        # x.append(np.mean([np.sum(d["tags"].item()) for _, d in traj.groupby(["branch.id"])]))
    x, y = np.array(y), np.array(x)
    x[np.isnan(x)] = np.nanmedian(x)
    y[np.isnan(y)] = np.nanmedian(y)
    # indexes = x != x.max()
    # x, y = x[indexes], y[indexes]
    # for method in [kendalltau, pearsonr, spearmanr]:
    #     print(method(x, y))
    print(np.sum(y <= 0.05) / 100)
    # plt.scatter(x, y)
    # plt.hist(y)
    # plt.show()

    beta_1, beta_0, rvalue, pvalue, _ = linregress(x, y)
    x_temp = np.sort(x)[:-1]
    y_hat = beta_0 + (beta_1 * x_temp)
    p = f"p={round(pvalue, 4)}" if pvalue > 0.001 else "p<0.001"
    # plt.plot(x_temp, y_hat, color="red", linewidth=2, label=f"r={round(rvalue, 3)}, {p}")
    # coeffs = np.polyfit(x, y, 2)
    # poly = np.poly1d(coeffs)
    # x_fit = np.linspace(x.min(), x.max(), 200)
    # y_fit = poly(x_fit)
    X = np.column_stack((x, x ** 2))
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    # print(model.summary())
    # X3 = np.column_stack((x, x ** 2, x ** 3))
    X3 = sm.add_constant(x)
    model_cubic = sm.OLS(y, X3).fit()
    print(model.compare_f_test(model_cubic))
    models = {d: fit_poly(x, y, d) for d in range(1, 6)}
    for d, m in models.items():
        print(d, m.aic, m.bic)
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    for d in range(1, 6):
        X_poly = PolynomialFeatures(d).fit_transform(x.reshape(-1, 1))
        scores = cross_val_score(LinearRegression(), X_poly, y, cv=5)
        print(d, scores.mean())
    # degree = min(models.items(), key=lambda m: m[1].bic)[0]
    # coeffs = np.polyfit(x, y, degree)
    # poly = np.poly1d(coeffs)
    # y_fit = poly(x_fit)
    # plt.plot(x_fit, y_fit, color="orange", label="Quadratic fit", linewidth=2)
    # logit_model = smf.logit("selfrep ~ phi", data=pd.DataFrame({"phi": x, "selfrep": y})).fit()
    # print(logit_model.summary())
    # pred = logit_model.predict({"phi": x_fit})
    # plt.plot(x_fit, pred, color="green", label="Logistic fit", linewidth=2)
    # plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    # plt.xlabel("$\\Phi$", fontsize=15)
    # plt.ylabel("Self-replication probability", fontsize=15)
    # plt.ylim(0.92, 1.01)
    # plt.legend()
    # plt.title("Intervention Experiment:\none dot per run", weight="bold", fontsize=20)
    # plt.tight_layout()
    # plt.savefig("figures/intervention.png")
    # plt.close()
    # plt.show()
    # exit()

    # all_data["phi"] = all_data.apply(lambda row: np.nanmedian(row["info"]), axis=1)
    # all_data["phi"] = all_data.apply(lambda row: np.nanmedian(info[row["mother.id"]]), axis=1)
    # all_data.dropna(axis=0, how="any", inplace=True)
    # print(all_data.shape)
    # all_data["phi"] = (all_data["phi"] - all_data["phi"].mean()) / all_data["phi"].std()
    # all_data["prob"] = (all_data["prob"] - all_data["prob"].mean()) / all_data["prob"].std()
    independent, dependent = "phi", "prob"
    model = smf.mixedlm(f"{independent} ~ {dependent} + (1 | gen)", data=all_data, groups=all_data["seed"],
                        # re_formula=f"~{dependent}"
                        ).fit()
    print(f"{dependent} slope: {model.params[dependent]} p-value: {model.pvalues[dependent]}")
    independent, dependent = "prob", "phi"
    model = smf.mixedlm(f"{independent} ~ {dependent} + (1 | gen)", data=all_data, groups=all_data["seed"],
                        # re_formula=f"~{dependent}"
                        ).fit()
    print(f"{dependent} slope: {model.params[dependent]} p-value: {model.pvalues[dependent]}")

    # grid for predictions
    phi_grid = np.linspace(all_data["phi"].min(), all_data["phi"].max(), 200)
    pred_df = all_data.iloc[:1].copy()  # template row
    pred_df = pred_df.loc[pred_df.index.repeat(len(phi_grid))].copy()
    pred_df["phi"] = phi_grid
    mean = model.predict(pred_df)
    # design matrix
    # X = np.column_stack([np.ones_like(phi_grid), phi_grid])
    # exog = model.model.exog_fe  # original design matrix
    design_info = model.model.data.design_info
    import patsy
    X = patsy.build_design_matrices([design_info], pred_df)[0]
    # beta = model.fe_params.values
    cov = model.cov_params().loc[model.fe_params.index, model.fe_params.index].values
    # prediction + SE
    # mean = X @ beta
    var = np.sum((X @ cov) * X, axis=1)
    se = np.sqrt(var)
    z = norm.ppf(0.975)
    lower = mean - z * se
    upper = mean + z * se

    ymin, ymax = all_data["prob"].min(), all_data["prob"].nlargest(2).iloc[-1]
    # mask = (mean >= ymin) & (mean <= ymax)
    phi_plot = phi_grid  # [mask]
    mean_plot = mean  # [mask]
    lower_plot = lower  # [mask]
    upper_plot = upper  # [mask]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(all_data["phi"], all_data["prob"], alpha=0.15, s=15)
    ax.plot(phi_plot, mean_plot, linewidth=3)
    ax.fill_between(phi_plot, lower_plot, upper_plot, alpha=0.2)

    # fixed-effect prediction
    # beta0 = model.fe_params["Intercept"]
    # beta1 = model.fe_params["phi"]
    # yhat = beta0 + beta1 * phi_grid
    # print(yhat.shape, phi_grid.shape, (y_hat <= all_data["prob"].max()).shape)

    # fig, ax = plt.subplots(figsize=(7, 5))
    # ax.scatter(all_data["phi"], all_data["prob"], alpha=0.15, s=15)
    # ax.plot(phi_grid[y_hat <= all_data["prob"].max()], yhat[y_hat <= all_data["prob"].max()], linewidth=3)
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(ymin - 0.01, ymax + 0.01)

    ax.set_xlabel("Phi")
    ax.set_ylabel("Self-replication (%)")
    ax.set_title("Mixed-effects fit: prob ~ Phi + (1|mother)")

    plt.show()


def downsample_mean(x, m):
    n = len(x)
    edges = np.linspace(0, n, m + 1, dtype=int)
    return np.array([
        x[edges[i]:edges[i + 1]].mean()
        for i in range(m)
    ])


def plot_phi_vs_prob(exp="output", measure="emergence", method=spearmanr):
    # tags = load_mat_types(binary=True)
    # data = read_info(directory=exp)[measure]
    old_data = load_all_mat_data()
    fig, axes = plt.subplots(figsize=(16, 5), nrows=1, ncols=1, sharey=True)
    if not isinstance(axes, list):
        axes = [axes]
    probs = []
    for color, directory in zip(["green", "red", "blue"],
                                ["base_causality", "new_causality_max", "new_causality_min"]):
        # x, y = [], []
        # for s, run in data.items():
        #     phi, c = run.flatten(), tags[s][:-1].flatten()
        #     if len(phi) != len(c):
        #         continue
        # if np.all(c == c[0]):
        #     c += np.random.normal(loc=0.0, scale=1e-6, size=len(c))
        #     y.append(np.nanmean(phi))
        #     x.append(np.nansum(c))
        # res = method(phi, downsample_mean(x=c, m=len(phi)))
        # x.append(res.statistic)
        # y.append(res.pvalue)
        # x, y = np.array(y), np.array(x)
        # x = StandardScaler().fit_transform(x.reshape(-1, 1)).ravel()
        # y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
        # print(np.mean(x <= 0.05))

        # axes[0].scatter(x, y, color="blue")
        # axes[0].set_title("Original runs", fontsize=15)
        # beta_1, beta_0, rvalue, pvalue, _ = linregress(x, y)
        # y_hat = beta_0 + (beta_1 * x)
        # p = f"p={round(pvalue, 4)}" if pvalue > 0.001 else "p<0.001"
        # axes[0].plot(x, y_hat, color="blue", linewidth=2, label=f"$\\rho$={round(rvalue, 3)}, {p}")

        # df = pd.DataFrame({"phi": x, "prob": y, "seed": np.arange(len(x))})
        # independent, dependent = "phi", "prob"
        # model = smf.mixedlm(f"{independent} ~ {dependent}", data=df, groups=df["seed"],
        # re_formula=f"~{dependent}"
        #                     ).fit()
        # print(f"{dependent} slope: {model.params[dependent]} p-value: {model.pvalues[dependent]}")
        # independent, dependent = "prob", "phi"
        # model = smf.mixedlm(f"{independent} ~ {dependent}", data=df, groups=df["seed"],
        # re_formula=f"~{dependent}"
        #                     ).fit()
        # print(f"{dependent} slope: {model.params[dependent]} p-value: {model.pvalues[dependent]}")

        metadata = load_data(directory=directory)
        data = load_all_mat_data_causality(directory=directory)
        metadata.replace([np.inf, -np.inf], np.nan, inplace=True)
        all_data = pd.merge(metadata, data, on=["seed"], how="inner")
        # all_data = all_data[all_data["n_sum"] != 0]
        all_data.dropna(axis=0, how="any", inplace=True)

        x, y = [], []
        for (seed,), traj in all_data.groupby(["seed"]):
            a, b = traj["phi"].values, traj["tags"].values[0][old_data[seed + 1].shape[1] // 2:].astype(np.float32)
            # if np.all(b == b[0]):
            #     b += np.random.normal(loc=0.0, scale=1e-6, size=len(b))
            x.append(np.nanmean(a))
            y.append(np.nanmean(b))
            # res = method(a, downsample_mean(x=b, m=len(a)))
            # x.append(res.pvalue)
            # y.append(res.statistic)
        x, y = np.array(x), np.array(y)
        probs.append(y)  # [x <= 0.05])
        x[np.isnan(x)] = np.nanmedian(x)
        y[np.isnan(y)] = np.nanmedian(y)
        print(np.mean(x <= 0.05))
        # indexes = x != x.max()
        # x, y = x[indexes], y[indexes]
        x = StandardScaler().fit_transform(x.reshape(-1, 1)).ravel()
        y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
        axes[0].scatter(x, y, color=color, alpha=0.5)
        axes[0].scatter(np.median(x), np.median(y), color=color, s=100)
        axes[0].set_title("With $\\Phi$ maximization", fontsize=15)

        beta_1, beta_0, rvalue, pvalue, _ = linregress(x, y)
        x_temp = np.sort(x)[:-1]
        y_hat = beta_0 + (beta_1 * x_temp)
        p = f"p={round(pvalue, 4)}" if pvalue > 0.001 else "p<0.001"
        axes[0].plot(x_temp, y_hat, color=color, linewidth=2, label=f"$\\rho$={round(rvalue, 3)}, {p}")

        df = pd.DataFrame({"phi": x, "prob": y, "seed": np.arange(len(x))})
        # df["phi"] = (df["phi"] - df["phi"].mean()) / df["phi"].std()
        # df["prob"] = (df["prob"] - df["prob"].mean()) / df["prob"].std()
        independent, dependent = "phi", "prob"
        model = smf.mixedlm(f"{independent} ~ {dependent}", data=df, groups=df["seed"],
                            # re_formula=f"~{dependent}"
                            ).fit()
        print(f"{dependent} slope: {model.params[dependent]} p-value: {model.pvalues[dependent]}")
        independent, dependent = "prob", "phi"
        model = smf.mixedlm(f"{independent} ~ {dependent}", data=df, groups=df["seed"],
                            # re_formula=f"~{dependent}"
                            ).fit()
        print(f"{dependent} slope: {model.params[dependent]} p-value: {model.pvalues[dependent]}")
        # probs.append(y)  # [x <= 0.05])

    for i, p1 in enumerate(probs):
        for j, p2 in enumerate(probs):
            if i == j:
                continue
            print(i, j,
                  # wilcoxon(p1, p2),
                  mannwhitneyu(p1, p2, alternative="greater"),
                  # _cliffs_delta(p1, p2)
                  )

    for ax in axes:
        # ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_xlabel("$\\Phi$", fontsize=15)
        ax.set_ylabel("Self-replication probability", fontsize=15)
        ax.legend()
    # fig.suptitle("Driving $\\Phi$ up increased self-replication probability", weight="bold", fontsize=20)
    # fig.tight_layout()
    plt.savefig("figures/phi_vs_prob.png")
    plt.close()

    # plt.figure(figsize=(8, 5))
    # plt.boxplot([probs[0], probs[1], probs[2]])
    # plt.xticks([1, 2, 3], ["Original runs", "With $\\Phi$ maximization", "With $\\Phi$ minimization"],
    #            fontsize=15)
    # plt.ylabel("Self-replication persistence\n(n. molecular steps)", fontsize=15)
    # y, hook = 1050, 50
    # plt.plot([1, 1, 2, 2], [y, y + hook, y + hook, y], lw=1.5, c="black")
    # plt.text(1.5, y + hook // 1.5, "***", ha="center", va="bottom", fontsize=20)
    # plt.ylim(-50, 1200)
    # plt.show()


def plot_causality_mixedlm(directory1="new_causality_max", directory2="new_causality_min",
                           directory3="base_causality"):
    fig, axes = plt.subplots(figsize=(24, 5), nrows=1, ncols=3)
    probs = []
    old_data = load_all_mat_data()
    for i, directory in enumerate([directory3, directory1, directory2]):
        metadata = load_data(directory=directory)
        data = load_all_mat_data_causality(directory=directory)
        metadata.replace([np.inf, -np.inf], np.nan, inplace=True)
        metadata = metadata.ffill().bfill()
        # metadata.replace([np.inf], np.nanmedian(metadata["phi"]), inplace=True)
        # metadata.replace([-np.inf], np.nanmedian(metadata["phi"]), inplace=True)
        all_data = pd.merge(metadata, data, on=["seed"], how="inner")
        all_data["prev_step"] = all_data["step"].shift(1).fillna(0).astype(np.int32)
        # all_data = all_data[all_data["n_sum"] != 0]
        all_data["tags"] = all_data.apply(lambda row: row["tags"][
                                                      row["prev_step"] + old_data[row["seed"] + 1].shape[1] // 2: row[
                                                                                                                      "step"] +
                                                                                                                  old_data[
                                                                                                                      row[
                                                                                                                          "seed"] + 1].shape[
                                                                                                                      1] // 2],
                                          axis=1)
        all_data = all_data[all_data["tags"].apply(lambda x: len(x) > 0)]
        all_data["prob"] = all_data.apply(lambda row: np.nansum(row["tags"]), axis=1)
        # all_data["prob"] = all_data.apply(lambda row: find_earliest_component(tags=row["tags"].astype(np.int32)), axis=1)
        # all_data.dropna(axis=0, how="any", inplace=True)

        # all_data["phi"] = (all_data["phi"] - all_data["phi"].mean()) / all_data["phi"].std()
        # all_data["prob"] = (all_data["prob"] - all_data["prob"].mean()) / all_data["prob"].std()
        independent, dependent = "phi", "prob"
        model = smf.mixedlm(f"{independent} ~ {dependent} + C(gen)", data=all_data, groups=all_data["seed"],
                            # re_formula=f"~{dependent}"
                            ).fit()
        print(f"{dependent} slope: {model.params[dependent]} p-value: {model.pvalues[dependent]}")
        independent, dependent = "prob", "phi"
        model = smf.mixedlm(f"{independent} ~ {dependent} + C(gen)", data=all_data, groups=all_data["seed"],
                            # re_formula=f"~{dependent}"
                            ).fit()
        print(f"{dependent} slope: {model.params[dependent]} p-value: {model.pvalues[dependent]}")
        probs.append(all_data["prob"].values)

        # grid for predictions
        phi_grid = np.linspace(all_data["phi"].min(), all_data["phi"].max(), 200)
        # pred_df = all_data.iloc[:1].copy()  # template row
        # pred_df = pred_df.loc[pred_df.index.repeat(len(phi_grid))].copy()
        # pred_df["phi"] = phi_grid
        # mean = model.predict(pred_df)
        # design matrix
        # design_info = model.model.data.design_info
        # import patsy
        # X = patsy.build_design_matrices([design_info], pred_df)[0]
        # cov = model.cov_params().loc[model.fe_params.index, model.fe_params.index].values
        # prediction + SE
        # var = np.sum((X @ cov) * X, axis=1)
        # se = np.sqrt(var)
        # z = norm.ppf(0.95)
        # lower = mean - z * se
        # upper = mean + z * se

        ymin, ymax = all_data["prob"].min(), all_data["prob"].nlargest(2).iloc[-1]
        ax = axes[i]
        color = "red"
        if i == 0:
            color = "green"
        elif i == 2:
            color = "blue"
        ax.scatter(all_data["phi"], all_data["prob"], color=color, alpha=0.05, s=15)
        # h = ax.hist2d(all_data["phi"], all_data["prob"], bins=25, cmap='viridis')
        # plt.colorbar(h[3], ax=ax, label='Counts')
        # p = f"p={round(model.pvalues[dependent], 4)}" if model.pvalues[dependent] > 0.001 else "p<0.001"
        # ax.plot(phi_grid, mean, color=color, linewidth=3, label=f"Mixed-effects fit ± 95% CI\n$slope_\\Phi$={round(model.params[dependent], 4)}, {p}")
        # ax.fill_between(phi_grid, lower, upper, color=color, alpha=0.2)
        beta_1, beta_0, rvalue, pvalue, _ = linregress(all_data["phi"], all_data["prob"])
        y_hat = beta_0 + (beta_1 * phi_grid)
        p = f"p={round(pvalue, 4)}" if pvalue > 0.001 else "p<0.001"
        ax.plot(phi_grid, y_hat, color=color, linewidth=3, label=f"r={round(rvalue, 3)}, {p}")
        d = np.abs(ymax - ymin) / 25
        ax.set_ylim(ymin - d, ymax + d)

        ax.set_xlabel("$\\Phi$ (standardized)", fontsize=15)
        # if i == 0:
        ax.set_ylabel("Self-replication persistence\n(n. molecular steps, standardized)", fontsize=15)
        ax.legend()
        title = "Original runs" if "base" in directory else f"With ${directory.split('_')[2]}imization$ of $\\Phi$"
        ax.set_title(title, fontsize=15)
    fig.suptitle("Intervening on $\\Phi$ drove self-replication persistence", weight="bold", fontsize=20)
    # fig.tight_layout()
    plt.savefig("figures/new_intervention.pdf")
    plt.savefig("figures/new_intervention.png")
    plt.close()

    for i, p1 in enumerate(probs):
        for j, p2 in enumerate(probs):
            if i == j:
                continue
            print(i, j, mannwhitneyu(p1, p2, alternative="greater"), _cliffs_delta(p1, p2))

    plt.violinplot(probs)
    plt.show()


def plot_causality_mixedlm_pretty(directory1="new_causality_max", directory2="new_causality_min",
                                  directory3="base_causality", width=0.25):
    probs = []
    old_data = load_all_mat_data()
    for i, directory in enumerate([directory3, directory1, directory2]):
        metadata = load_data(directory=directory)
        data = load_all_mat_data_causality(directory=directory)
        metadata.replace([np.inf, -np.inf], np.nan, inplace=True)
        # metadata = metadata.ffill().bfill()
        all_data = pd.merge(metadata, data, on=["seed"], how="inner")
        mask_zero_or_after = all_data.groupby("seed")["n_sum"].transform(lambda s: s.eq(0).cummax())
        all_data = all_data[~mask_zero_or_after]
        # all_data = all_data[all_data["gen"] >= 25]
        all_data["prev_step"] = all_data["step"].shift(1).fillna(0).astype(np.int32)
        # all_data = all_data[all_data["n_sum"] != 0]
        all_data["tags"] = all_data.apply(lambda row: row["tags"][
                                                      row["prev_step"] + old_data[row["seed"] + 1].shape[1] // 2: row[
                                                                                                                      "step"] +
                                                                                                                  old_data[
                                                                                                                      row[
                                                                                                                          "seed"] + 1].shape[
                                                                                                                      1] // 2],
                                          axis=1)
        # all_data["tags"] = all_data.apply(lambda row: row["tags"][row["prev_step"]: row["step"]],
        #                                   axis=1)
        all_data = all_data[all_data["tags"].apply(lambda x: len(x) > 0)]
        # all_data["prob"] = all_data.apply(lambda row: np.nansum(row["tags"].astype(np.int32)),
        #                                   axis=1)
        all_data["prob"] = all_data.apply(lambda row: np.nanmean(row["tags"].astype(np.int32)),
                                          axis=1)
        # all_data["prob"] = all_data.apply(lambda row: find_earliest_component(tags=row["tags"]), axis=1)
        all_data.dropna(axis=0, how="any", inplace=True)
        # probs.append(all_data["prob"].values)
        probs.append([(seed, traj["prob"].mean() > 0.5) for (seed,), traj in all_data.groupby(["seed"])])
        # print(directory, np.mean([traj["prob"].mean() for (seed,), traj in all_data.groupby(["seed"])]))
        plt.bar(i - width / 2,
                # (all_data["prob"] > 0.5).mean(),
                all_data["prob"].mean(),
                yerr=(all_data["prob"] > 0.5).std(),
                width=width,
                alpha=0.7,
                capsize=5,
                color="red",
                label="self-replication" if i == 0 else None)
        plt.bar(i + width / 2,
                # (all_data["prob"] <= 0.5).mean(),
                all_data["prob"].mean(),
                yerr=(all_data["prob"] <= 0.5).std(),
                width=width,
                alpha=0.7,
                capsize=5,
                color="blue",
                label="drift" if i == 0 else None)
    plt.savefig("figures/prob_bars.png")
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.violinplot([[y for x, y in probs[0]], [y for x, y in probs[1]]])
    plt.xticks([1, 2], ["Original runs", "$\\Phi$ maximization"],
               fontsize=15)
    # plt.gca().yaxis.set_major_formatter(
    #     mtick.FuncFormatter(lambda y, _: f"{y * 100:g}")
    # )
    plt.ylabel("Self-replication persistence\n(n. molecular steps)", fontsize=15)
    # y, hook = 1050, 50
    # plt.plot([1, 1, 2, 2], [y, y + hook, y + hook, y], lw=1.5, c="black")
    # plt.text(1.5, y + hook // 1.5, "***", ha="center", va="bottom", fontsize=20)
    # plt.ylim(-50, 1200)
    # plt.xticks([0, 1, 2], ["Original runs", "$\\Phi$ maximization", "$\\Phi$ minimization"], fontsize=15)
    # plt.legend()
    plt.show()

    for i, p1 in enumerate(probs):
        for j, p2 in enumerate(probs):
            if i == j:
                continue
            p1 = pd.DataFrame(p1, columns=["s", "prob1"])
            p2 = pd.DataFrame(p2, columns=["s", "prob2"])
            merged = pd.merge(p1, p2, on="s")
            print(i, j,
                  ttest_rel(merged["prob1"], merged["prob2"], alternative="greater"),
                  wilcoxon(merged["prob1"], merged["prob2"], alternative="greater"),
                  mannwhitneyu(merged["prob1"], merged["prob2"], alternative="greater"),
                  _cliffs_delta(merged["prob1"], merged["prob2"]))
            diff = merged["prob1"] - merged["prob2"]
            print("Difference:", merged["prob1"].mean() - merged["prob2"].mean())
            print("Mean paired improvement:", diff.mean())
            print("Runs improved:", np.sum(diff > 0))
            print("Runs worsened:", np.sum(diff < 0))
            print("Runs unchanged:", np.sum(diff == 0))
            print(wilcoxon(diff, alternative="greater"), )


def plot_causality_bars(method=spearmanr):
    all_data = pd.DataFrame(columns=["seed", "phi", "prob", "exp"])
    old_data = load_all_mat_data()
    for directory in ["base_causality", "new_causality_max", "new_causality_min"]:
        metadata = load_data(directory=directory)
        data = load_all_mat_data_causality(directory=directory)
        metadata.replace([np.inf, -np.inf], np.nan, inplace=True)
        # metadata.replace([np.inf], np.nanmax(metadata["phi"]), inplace=True)
        # metadata.replace([-np.inf], np.nanmin(metadata["phi"]), inplace=True)
        curr_data = pd.merge(metadata, data, on=["seed"], how="inner")
        curr_data["exp"] = directory
        # curr_data["prev_step"] = curr_data["step"].shift(1).fillna(0).astype(np.int32)
        # curr_data = curr_data[curr_data["n_sum"] != 0]
        # curr_data["tags"] = curr_data.apply(lambda row: row["tags"][row["prev_step"]: row["step"]],
        #                                     axis=1)
        curr_data = curr_data[curr_data["tags"].apply(lambda x: len(x) > 0)]
        curr_data.dropna(axis=0, how="any", inplace=True)
        for (seed,), traj in curr_data.groupby(["seed"]):
            a, b = traj["phi"].values, traj["tags"].values[0][old_data[seed + 1].shape[1] // 2:].astype(np.float32)
            # b = find_earliest_component(tags=b)
            # a = np.nan_to_num(a, posinf=np.max(a[np.isfinite(a)]), neginf=np.min(a[np.isfinite(a)]))
            # if len(a) < 2 or len(b) < 2:
            #     continue
            # elif np.all(b == b[0]):
            #     b += np.random.normal(loc=0.0, scale=1e-6, size=len(b))
            # res = method(a, downsample_mean(x=b, m=len(a)))
            all_data = pd.concat([all_data, pd.DataFrame([{"seed": seed,
                                                           "phi": np.nanmedian(a),
                                                           "prob": np.nanmean(b),
                                                           "exp": directory}])],
                                 ignore_index=True)

    new_data = pd.DataFrame()
    lines = []
    for (seed,), traj in all_data.groupby(["seed"]):
        if len(traj) < 3:
            continue
        new_data = pd.concat([new_data, pd.DataFrame([{"seed": seed,
                                                       "pbase": traj[traj["exp"] == "base_causality"]["prob"].item(),
                                                       "pmax": traj[traj["exp"] == "new_causality_max"]["prob"].item(),
                                                       "pmin": traj[traj["exp"] == "new_causality_min"]["prob"].item()
                                                       }])])
        lines.append([traj[traj["exp"] == "new_causality_max"]["prob"].item(),
                      traj[traj["exp"] == "new_causality_min"]["prob"].item()])
        plt.plot([1, 2], [traj[traj["exp"] == "base_causality"]["prob"].item(),
                          traj[traj["exp"] == "new_causality_max"]["prob"].item()])
        plt.plot([3, 4], [traj[traj["exp"] == "new_causality_max"]["prob"].item(),
                          traj[traj["exp"] == "new_causality_min"]["prob"].item()])

    for exp1 in ["pbase", "pmax", "pmin"]:
        for exp2 in ["pbase", "pmax", "pmin"]:
            if exp1 == exp2:
                continue
            if exp1 == "pbase" or (exp1 == "pmax" and exp2 == "pmin"):
                print(exp1, exp2,
                      wilcoxon(new_data[exp1], new_data[exp2]), np.median(new_data[exp1] - new_data[exp2]),
                      mannwhitneyu(new_data[exp1], new_data[exp2], alternative="less"),
                      _cliffs_delta(new_data[exp1], new_data[exp2]))
            name = "->".join([exp2, exp1])
            x = (new_data[exp1] - new_data[exp2]) / (new_data[exp2] + 1e-6)
            # q1 = np.percentile(x, 25)
            # q3 = np.percentile(x, 75)
            # iqr = q3 - q1
            # lower = q1 - 1.5 * iqr
            # upper = q3 + 1.5 * iqr
            # filtered = x[(x >= lower) & (x <= upper)]
            new_data[name] = x
            if name.startswith("pbase->"):
                print(name, wilcoxon(x), np.median(x), np.sum(x > 0.0) / len(x))

    plt.plot([1, 2], np.median(lines, axis=0), linewidth=10)
    plt.show()
    plt.close()
    cols = ["pbase->pmax", "pbase->pmin", "pmax->pmin",
            "pmin->pmax"]  # [col for col in new_data.columns if "->" in col]
    plt.boxplot(new_data[cols])
    plt.xticks(np.arange(1, len(cols) + 1), cols)
    # plt.ylim(-0.1, 2.5)
    plt.show()


def plot_optim_fitness(directory="causality", gen_var="gen", fit_var="phi"):
    data = load_data(directory=directory)
    data = data.replace([np.inf, -np.inf], np.nan)
    data.dropna(how="any", inplace=True)
    fig, axes = plt.subplots(figsize=(20, 5), nrows=1, ncols=len(data["exp"].unique()), sharey=True)
    for col, ((exp,), traj) in enumerate(data.groupby(["exp"])):
        if "trace" in traj.columns:
            per_trace = traj.groupby(["trace", gen_var])[fit_var].agg(["median"]).reset_index()
        else:
            traj["median"] = traj[fit_var]
            per_trace = traj
        median = per_trace.groupby(gen_var).agg({"median": ["median"]})
        axes[col].plot(median[("median", "median")])
        err = per_trace.groupby(gen_var).agg({"median": ["std"]})
        axes[col].fill_between(np.arange(len(median)),
                               median[("median", "median")] - err[("median", "std")],
                               median[("median", "median")] + err[("median", "std")],
                               alpha=0.25)
        axes[col].set_xlabel("Generations", fontsize=15)
        axes[col].set_ylabel("$\\Phi$", fontsize=15)
        axes[col].set_title(f"{exp} $\\Phi$", fontsize=15)
    plt.savefig("figures/new_fitness.png")
    plt.close()


def plot_optim_boxplot(directory="optim"):
    data = load_train_tags(path=os.path.join(directory, "test"))
    data.dropna(how="any", inplace=True)
    fig, axes = plt.subplots(figsize=(20, 5), nrows=1, ncols=4)
    for phase in ["relax", "train", "test"]:
        data[f"{phase}_agg"] = data.apply(lambda row: number_of_runs(row[f"{phase}.tags"]),
                                          # find_earliest_component(tags=row[f"{phase}.tags"]),
                                          # keep_largest_component(row=row[f"{phase}.tags"]).mean(),  # row[f"{phase}.tags"].mean(),
                                          axis=1)
        data[f"long.{phase}_agg"] = data.apply(lambda row: number_of_runs(row[f"long.{phase}.tags"]),
                                               # find_earliest_component(tags=row[f"long.{phase}.tags"]),
                                               # keep_largest_component(row=row[f"long.{phase}.tags"]).mean(),  # row[f"long.{phase}.tags"].mean(),
                                               axis=1)
    data_max = data[data["exp"] == "max"]
    data_min = data[data["exp"] == "min"]
    data_base = data[(data["exp"] == "max") & (data["is_rogue"] == False)].groupby("seed.sim", as_index=False).first()
    axes[0].boxplot([data_max["relax_agg"],
                     data_max["train_agg"],
                     data_max["test_agg"]])
    axes[1].boxplot([data_min["relax_agg"],
                     data_min["train_agg"],
                     data_min["test_agg"]])
    axes[2].boxplot([data_base["long.relax_agg"],
                     data_base["long.train_agg"],
                     data_base["long.test_agg"]])
    print(linregress(data_max["fitness.delta"], data_max["test_agg"]))
    print(linregress(data_min["fitness.delta"], data_min["test_agg"]))
    print(linregress(data_base["fitness.delta"], data_base["long.test_agg"]))
    for ax, title in zip(axes, ["max", "min", "base"]):
        ax.set_xticks([1, 2, 3], ["relax", "train", "test"])
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y * 100:g}"))
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=15)
    phase = "test"
    test = mannwhitneyu
    print(test(data_base[f"long.{phase}_agg"], data_max.groupby("seed.sim")[f"{phase}_agg"].mean(),
               alternative="greater"),
          _cliffs_delta(data_base[f"long.{phase}_agg"], data_max.groupby("seed.sim")[f"{phase}_agg"].mean()),
          test(data_base[f"long.{phase}_agg"], data_min.groupby("seed.sim")[f"{phase}_agg"].mean(),
               alternative="greater"),
          _cliffs_delta(data_base[f"long.{phase}_agg"], data_min.groupby("seed.sim")[f"{phase}_agg"].mean()),
          test(data_max.groupby("seed.sim")[f"{phase}_agg"].mean(),
               data_min.groupby("seed.sim")[f"{phase}_agg"].mean(),
               alternative="less"),
          _cliffs_delta(data_max.groupby("seed.sim")[f"{phase}_agg"].mean(),
                        data_min.groupby("seed.sim")[f"{phase}_agg"].mean()))

    model = smf.mixedlm(
        f"{phase}_agg ~ exp",
        data=data,
        groups=data["seed.sim"]
    ).fit()
    print(model.summary())
    plt.savefig("figures/optim_boxplot.png")
    plt.close()

    plt.boxplot([data_max["test_agg"], data_base["long.test_agg"], data_min["test_agg"]])
    plt.ylabel("Self-replication prob. during TEST\n(longest replicator)", fontsize=15)
    plt.xticks([1, 2, 3], ["max $\\Phi$", "base", "min $\\Phi$"], fontsize=15)
    y, hook = 0.5, 0.05
    plt.plot([1, 1, 1.9, 1.9], [y, y + hook, y + hook, y], lw=1.5, c="black")
    plt.text(1.5, y + hook, "***", ha="center", va="bottom", fontsize=20)
    plt.plot([2.1, 2.1, 3, 3], [y, y + hook, y + hook, y], lw=1.5, c="black")
    plt.text(2.5, y + hook, "***", ha="center", va="bottom", fontsize=20)
    y = 0.7
    plt.plot([1, 1, 3, 3], [y, y + hook, y + hook, y], lw=1.5, c="black")
    plt.text(2, y + hook, "ns", ha="center", va="bottom", fontsize=20)
    plt.savefig("figures/optim_boxplot.png")
    plt.close()


def plot_new_causality_analysis(directory="causality", binary=True, by_time=True):
    data = load_causality_tags(path=directory, by_time=by_time)
    if binary:
        data["tags"] = data.apply(lambda row: np.where(row["tags"] >= 1, 1, 0), axis=1)
    data["phi_agg"] = data.apply(lambda row: row["tags"][int(len(row["tags"]) * 0.0):].mean(),
                                 # number_of_runs(row["tags"]),
                                 # find_earliest_component(row=row["tags"][int(len(row["tags"]) * 0.75):], by_type=False, keep_largest=True),
                                 # keep_largest_component(row=row["tags"], by_type=True).mean(),  # row[f"{phase}.tags"].mean(),
                                 axis=1)
    data_max = data[data["exp"] == "max"]
    data_min = data[data["exp"] == "min"]
    data_base = data[data["exp"] == "base"]
    if by_time:
        for label, d in zip(["max", "base", "min"], [data_max, data_base, data_min]):
            # median = d.groupby(d.i)["phi_agg"].median()
            # plt.plot(median, label=label)
            # err = d.groupby(d.i)["phi_agg"].std()
            # plt.fill_between(np.arange(len(median)), median - err, median + err, alpha=0.25)
            # plt.scatter(d["i"], d["phi_agg"], color=colors[label], alpha=0.05)
            beta_1, beta_0, rvalue, pvalue, _ = linregress(d["i"], d["phi_agg"])
            y_hat = beta_0 + (beta_1 * d["i"])
            p = f"p={round(pvalue, 4)}" if pvalue > 0.001 else "p<0.001"
            plt.plot(d["i"], y_hat, linewidth=3, label=f"{label} r={round(rvalue, 3)}, {p}")
            s_err = np.sqrt(np.sum((d["phi_agg"] - y_hat) ** 2) / (len(d["i"]) - 2))
            x_mean = d["i"].mean()
            Sxx = np.sum((d["i"] - x_mean) ** 2)
            tval = t.ppf(0.975, len(d["i"]) - 2)
            ci = tval * s_err * np.sqrt(1 / len(d["i"]) + (d["i"] - x_mean) ** 2 / Sxx)
            plt.fill_between(d["i"], y_hat - ci, y_hat + ci, alpha=0.25)
        plt.legend()
        plt.show()
        return
    print(linregress(data_max["phi"], data_max["phi_agg"]))
    print(linregress(data_min["phi"], data_min["phi_agg"]))
    print(linregress(data_base["phi"], data_base["phi_agg"]))
    test = mannwhitneyu
    print(test(data_base["phi_agg"], data_max["phi_agg"],
               alternative="less"),
          _cliffs_delta(data_base["phi_agg"], data_max["phi_agg"]),
          test(data_base["phi_agg"], data_min["phi_agg"],
               alternative="less"),
          _cliffs_delta(data_base["phi_agg"], data_min["phi_agg"]),
          test(data_max["phi_agg"], data_min["phi_agg"],
               alternative="greater"),
          _cliffs_delta(data_max["phi_agg"], data_min["phi_agg"]))

    # model = smf.mixedlm(
    #     "phi_agg ~ exp",
    #     data=data,
    #     groups=data["seed"]
    # ).fit()
    # print(f"exp slope: {model.params['exp']} p-value: {model.pvalues['exp']}")

    plt.boxplot([_filter_outliers(data_max["phi_agg"]),
                 _filter_outliers(data_base["phi_agg"]),
                 _filter_outliers(data_min["phi_agg"])])
    plt.ylabel("Self-replication persistence", fontsize=15)
    plt.xticks([1, 2, 3], ["max $\\Phi$", "base", "min $\\Phi$"], fontsize=15)
    y, hook = 1450, 50
    plt.plot([1, 1, 1.9, 1.9], [y, y + hook, y + hook, y], lw=1.5, c="black")
    plt.text(1.5, y + hook, "***", ha="center", va="bottom", fontsize=20)
    y = 1600
    plt.plot([1, 1, 3, 3], [y, y + hook, y + hook, y], lw=1.5, c="black")
    plt.text(2, y + hook, "***", ha="center", va="bottom", fontsize=20)
    y = 1050
    plt.plot([2.1, 2.1, 3, 3], [y, y + hook, y + hook, y], lw=1.5, c="black")
    plt.text(2.5, y + hook, "***", ha="center", va="bottom", fontsize=20)
    plt.ylim(250, 1800)
    plt.savefig("figures/new_optim_boxplot.png")
    plt.close()


def _add_bracket(y, hook, text, ax=plt):
    ax.plot([1, 1, 1.9, 1.9], [y, y + hook, y + hook, y], lw=1.5, c="black")
    ax.text(1.5, y + hook, text, ha="center", va="bottom", fontsize=20)


def _get_sig(pval):
    sig = "ns"
    if pval <= 0.05:
        sig = "*"
    if pval <= 0.01:
        sig = "**"
    if pval <= 0.001:
        sig = "***"
    return sig


def plot_new_causality_analysis_pretty(directory="old_causality", binary=True):
    data = load_causality_tags(path=directory, by_time=False)
    if binary:
        data["tags"] = data.apply(lambda row: np.where(row["tags"] >= 1, 1, 0), axis=1)
    data["Persistence\nmolecular steps"] = data.apply(lambda row: row["tags"].sum(), axis=1)
    data["Probability\n% of total molecular steps"] = data.apply(lambda row: row["tags"].mean(), axis=1)
    data["Consistency\nPearson's $\\rho$ of consecutive steps"] = data.apply(lambda row: number_of_runs(row["tags"]),
                                                                             axis=1)
    data["Time to first replicator\n% of total molecular steps"] = data.apply(
        lambda row: find_earliest_component(row=row["tags"][int(len(row["tags"]) * 0.75):],
                                            by_type=False, keep_largest=True), axis=1)
    data_max = data[data["exp"] == "max"]
    data_min = data[data["exp"] == "min"]
    data_base = data[data["exp"] == "control"]

    cols = [col for col in data.columns if "steps" in col]
    # fig, axes = plt.subplots(figsize=(20, 5), nrows=1, ncols=len(cols))
    for i, col in enumerate(cols):
        for label, d in zip(["max$\\Phi^r$", "control", "min$\\Phi^r$"], [data_max, data_base, data_min]):
            print(col.split("\n")[0], label, d[col].mean(), "±", d[col].std())
        # axes[i].boxplot([data.pivot(index="seed", columns="exp", values=col)
        #                 .assign(delta=lambda x: x[exp] - x["control"])
        #                 .reset_index()["delta"] for exp in ["max", "min"]])
        # axes[i].set_xticks([1, 2], ["max", "min"])
        # axes[i].set_title(col)
    # plt.show()
    # plt.close()

    fig = plt.figure(figsize=(20, 14), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    # gs = fig.add_gridspec(
    #     2, 2,
    #     height_ratios=[1.5, 1]
    # )

    ax1 = fig.add_subplot(gs[0, :])  # entire first row
    ax2 = fig.add_subplot(gs[1, 0])  # second row, left
    ax3 = fig.add_subplot(gs[1, 1])  # second row, right
    # fig, axes = plt.subplots(figsize=(20, 5), nrows=1, ncols=2)
    # for ax, col in zip(axes, ["Persistence\nmolecular steps",
    #                           "Probability\n% of total molecular steps",
    #                           "Consistency\nPearson's $\\rho$ of consecutive steps",
    #                           "Time to first replicator\n% of total molecular steps"]):
    import matplotlib.image as mpimg
    img1 = mpimg.imread("figures/intervention_diagram.png")
    ax1.imshow(img1)
    ax1.axis("off")
    ax1.set_title("A) Intervention experiment to drive $\\Phi^r$ up (or down) in GARD assemblies",
                  weight="bold",
                  fontsize=35)

    col = "Persistence\nmolecular steps"
    ax2.boxplot([_filter_outliers(data_max[col]),
                 _filter_outliers(data_base[col]),
                 _filter_outliers(data_min[col])])
    ax2.set_ylabel(col.split("\n")[1], fontsize=25)
    ax2.set_xticks([1, 2, 3], ["max $\\Phi^r$", "base", "min $\\Phi^r$"], fontsize=25)
    y, hook = 1450, 50
    ax2.plot([1, 1, 1.9, 1.9], [y, y + hook, y + hook, y], lw=1.5, c="black")
    ax2.text(1.5, y + hook, _get_sig(mannwhitneyu(data_max[col], data_base[col],
                                                  alternative="greater").pvalue),
             ha="center", va="bottom", fontsize=20)
    y = 1050
    ax2.plot([2.1, 2.1, 3, 3], [y, y + hook, y + hook, y], lw=1.5, c="black")
    ax2.text(2.5, y + hook, _get_sig(mannwhitneyu(data_base[col], data_min[col],
                                                  alternative="greater").pvalue), ha="center", va="bottom",
             fontsize=20)
    y = 1600
    ax2.plot([1, 1, 3, 3], [y, y + hook, y + hook, y], lw=1.5, c="black")
    ax2.text(2, y + hook, _get_sig(mannwhitneyu(data_max[col], data_min[col],
                                                alternative="greater").pvalue), ha="center", va="bottom",
             fontsize=20)
    ax2.set_ylim(250, 1800)
    ax2.set_title("B) Self-replication persistence increases\nwhen maximizing $\\Phi^r$", weight="bold",
                  fontsize=32)

    data = load_causality_tags(path=directory, by_time=True)
    if binary:
        data["tags"] = data.apply(lambda row: np.where(row["tags"] >= 1, 1, 0), axis=1)
    col = "Self-replication probability\n% of total molecular steps"
    data[col] = data.apply(lambda row: row["tags"].mean(), axis=1)
    data_max = data[data["exp"] == "max"]
    data_min = data[data["exp"] == "min"]
    data_base = data[data["exp"] == "control"]
    for label, d in zip(["max$\\Phi^r$", "control", "min$\\Phi^r$"], [data_max, data_base, data_min]):
        beta_1, beta_0, rvalue, pvalue, _ = linregress(d["i"], d[col])
        x_hat = np.linspace(d["i"].min(), d["i"].max(), 100)
        X = sm.add_constant(d["i"])
        model = sm.OLS(d[col], X).fit()
        X_fit = sm.add_constant(x_hat)
        pred = model.get_prediction(X_fit).summary_frame()
        p = f"p={round(pvalue, 4)}" if pvalue > 0.001 else "p<0.001"
        ax3.plot(x_hat, pred["mean"], linewidth=3)
        mask = np.isfinite(x_hat) & np.isfinite(pred["mean_ci_lower"]) & np.isfinite(pred["mean_ci_upper"])
        ax3.fill_between(
            x_hat[mask],
            pred["mean_ci_lower"][mask],
            pred["mean_ci_upper"][mask],
            label=f"{label} $\\rho$={round(rvalue, 3)}, {p}",
            alpha=0.2,
        )
    ax3.set_xlabel("GARD generations", fontsize=25)
    ax3.set_ylabel(col, fontsize=25)
    ax3.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda y_tick, _: f"{y_tick * 100:g}")
    )
    ax3.legend(fontsize=17.5)
    ax3.grid(True)
    ax3.set_title("C) Self-replication probability increases\nthrough max$\\Phi^r$ intervention",
                  weight="bold", fontsize=32.)

    plt.subplots_adjust(top=0.825)
    plt.savefig("figures/new_optim_boxplot_pretty_new.png")
    plt.close()


def plot_lifeless(directory="causality", params=("kf", "kb", "Nmax"), binary=True):
    params = list(params)
    data = load_lifeless(path=directory)
    n_params = data[params].drop_duplicates().shape[0]
    fig, axes = plt.subplots(figsize=(8 * n_params, 5), nrows=1, ncols=n_params)
    if binary:
        data["tags"] = data.apply(lambda row: np.where(row["tags"] >= 1, 1, 0), axis=1)
    data["agg"] = data.apply(lambda row: row["tags"].mean(), axis=1)

    for i, (param_vals, traj) in enumerate(data.groupby(params)):
        data_max = traj[traj["exp"] == "max"]
        data_min = traj[traj["exp"] == "min"]
        data_base = traj[traj["exp"] == "control"]
        axes[i].boxplot([data_max["agg"], data_min["agg"], data_base["agg"]])
        axes[i].set_xticks([1, 2, 3], ["max", "min", "base"])
        title = " ".join([f"{p}:{v}" for p, v in zip(params, param_vals)])
        axes[i].set_title(title, fontsize=15)
        print(f"{title}: {np.median(data_base['agg'])}")

        print(f"max vs. base: {mannwhitneyu(data_max['agg'], data_base['agg'], alternative='greater')}")
        print(f"min vs. base: {mannwhitneyu(data_min['agg'], data_base['agg'], alternative='greater')}")
        print(f"max vs. min: {mannwhitneyu(data_max['agg'], data_min['agg'], alternative='greater')}")

    plt.savefig("figures/lifeless.png")
    plt.close()


def plot_memory(exp="output", measure="emergence"):
    df = read_info(directory=exp)[measure]
    ps = []
    for s, phi in df.items():
        phi = phi.flatten()
        max_peaks = find_peaks(phi)[0]
        min_peaks = find_peaks(-phi)[0]
        new_phi = np.zeros_like(phi)
        new_phi[max_peaks] = 1
        new_phi[min_peaks] = 1
        p = acorr_ljungbox(np.diff(new_phi), lags=[2], return_df=True)["lb_pvalue"].iloc[0]
        ps.append(p)
    print(f"median ± std: {np.median(ps)}±{np.std(ps)}")
    print(f"% significant: {np.mean([p <= 0.05 for p in ps])}")
    plt.hist(ps)
    plt.show()


if __name__ == "__main__":
    # for file in os.listdir("lifeless/traces"):
    #     if "base" in file and "mat" in file:
    #         seed = int(file.split(".")[0])
    #         if seed <= 30:
    #             os.system(f"mv lifeless/traces/{file} causality/traces/")
    # exit()
    # for file in os.listdir("ami_data/traces"):
    #     if file.endswith("txt"):
    #         tags = loadmat(os.path.join("ami_data", "tags", file.replace("txt", "mat")))["tags"]
    #         data = pd.read_csv(os.path.join("ami_data", "traces", file), sep=";")
    #         data["composition"] = data.apply(lambda row: np.array([int(elem) for elem in row["composition"].split("/")]), axis=1)
    #         composition = np.vstack(data["composition"].values)
    #         assert tags.shape[1] == composition.shape[0]
    #         print(file, tags.shape, composition.shape)
    # exit()
    # for file in os.listdir("/Users/federicopigozzi/Downloads/ami_data/tags/"):
    #     if file.endswith("mat"):
    #         name = "/Users/federicopigozzi/Downloads/ami_data/tags/" + file
    #         os.rename(name, "/Users/federicopigozzi/Downloads/ami_data/tags/" + file.split(".")[0] + ".0,0.base.mat")
    # d = loadmat(os.path.join("ami_data/old_traces", file))["history"]
    # tags = loadmat(os.path.join("ami_data/tags", file))["tags"]
    # tags = np.where(tags >= 1, 1, 0)
    # savemat(os.path.join("ami_data/tags", file), {"tags": tags})
    # new_data = pd.DataFrame({"step": np.arange(len(d)),
    #                          "n_sum": d.sum(axis=1),
    #                          "composition": ["/".join([str(elem) for elem in _d]) for _d in d]})
    # new_data.to_csv("ami_data/traces/" + file.replace("mat", "txt"), sep=";", index=False)
    # print(tags.shape)
    # print(d["history"].shape)
    # for file in os.listdir("causality/traces"):
    #     trace = loadmat(f"causality/traces/{file}")
    #      print(trace["history"].shape)
    #     seed = int(file.split(".")[0])
    #     trace = loadmat(f"GARD-model/Matlab/SourceCode/Doron_Lancet_GARD_Next_generation/GARD_v10/data_long/GARD_run_seed_{seed + 1:03d}.mat")
    #     tags = loadmat("optim/test/" + file)
    #     print(trace["o"]["history"][0, 0].shape[1], tags["tags"].shape, tags["ttrain"][0][0], tags["ttest"][0][0])
    # exit()
    # d = load_all_mat_data(attribute="fluxes")
    # plot_new_molar_fractions(data=d[:10, :99, :])
    # exit()
    # d = load_data()
    # d = d[d["i"] < 500]
    # plot_info()
    # plot_samples()
    # check_noise()
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
    # plot_learning()
    # plot_learning_pretty()
    # plot_branching_analysis()
    # plot_branching_analysis_bis()
    # plot_causality_analysis()
    # plot_phi_vs_prob()
    # plot_causality_analysis(directory="new_new_causality_max")
    # plot_causality_analysis(directory="new_causality_min")
    # plot_causality_mixedlm()
    # plot_causality_mixedlm_pretty()
    # plot_causality_bars()
    # plot_optim_fitness()
    # plot_optim_boxplot()
    # plot_new_causality_analysis()
    # plot_new_causality_analysis_pretty()
    # plot_lifeless()
    plot_memory()
    # other_metrics()
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
