import os
from multiprocessing import Pool

import networkx.exception
import numpy as np
from scipy.io import loadmat
from scipy.stats import zscore

from information import mutual_information_matrix, minimum_information_bipartition, local_phi_id, local_phi_r, \
    preprocess_data
from plotting import load_data, load_mat_data, load_all_mat_data, load_all_mat_data_branching
from utils import set_seed, parse_args, MEASURES


def compute_sim_info(data):
    # data = preprocess_data(data=data)
    # print(np.sum(np.isnan(data)), np.sum(np.isinf(data)))
    # var_per_species = np.var(data, axis=1)
    # constant_species = np.where(var_per_species == 0)[0]
    # print(f"{len(constant_species)} constant species:", constant_species)
    # rank = np.linalg.matrix_rank(np.cov(data))
    # print(f"Covariance rank: {rank}/{data.shape[0]}")
    data = np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    data = data.astype(np.float64, copy=False)
    info = {}
    mi_mat = mutual_information_matrix(data, alpha=1, bonferonni=False, lag=1)
    mib = minimum_information_bipartition(mi_mat, noise=True)  # noise=False)
    component_1 = data[mib[0], :].mean(axis=0)
    component_2 = data[mib[1], :].mean(axis=0)
    data_reduced = np.vstack((component_1, component_2))
    phi_results = local_phi_id(0, 1, data_reduced)
    info["synergy"] = phi_results.nodes[(((0, 1),), ((0, 1),))]["pi"]
    info["causation"] = phi_results.nodes[(((0, 1),), ((0,),))]["pi"] + phi_results.nodes[(((0, 1),), ((1,),))]["pi"]
    info["redundancy"] = phi_results.nodes[(((0,), (1,)), ((0,), (1,)))]["pi"] + \
                         phi_results.nodes[(((0,), (1,)), ((0,), (1,)))]["pi"]
    info["integrated"] = local_phi_r(phi_results)
    info["emergence"] = np.nansum([info["synergy"], info["causation"]])  # info["synergy"] + info["causation"]
    return info


def compute_branch_info(args):
    idx, row, comp = args
    # branch_point = old_d.shape[1] // 2
    # comp = preprocess_data(data=row["trace"][:, :branch_point + row["trace"].shape[1] // 10])
    try:
        info = compute_sim_info(data=comp)
    except networkx.exception.NetworkXError:
        raise
    for measure in MEASURES:
        np.save(os.path.join("output_branching", measure.lower(), ".".join([str(row["mother.id"]),
                                                                            str(row["branch.id"])])), info[measure])
    print(idx)


def compute_long_info(file):
    s = int(file.split("_")[-1].split(".")[0])  # - 1
    train = loadmat(os.path.join("optim", "traces", file))["history"]
    test = loadmat(os.path.join("optim", "test", file))["history_test"]
    ttrain = loadmat(os.path.join("optim", "test", file))["ttrain"][0][0]
    relax = loadmat(f"GARD-model/Matlab/SourceCode/Doron_Lancet_GARD_Next_generation/GARD_v10/data_long/GARD_run_seed_{s + 1:03d}.mat")["o"]["history"][0][0]
    relax = relax[:, :ttrain]
    trace = np.hstack([relax, train, test])
    # long_trace = loadmat(os.path.join("GARD-model/Matlab/SourceCode/Doron_Lancet_GARD_Next_generation/GARD_v10/data_long", file))["o"]["history"][0][0]
    # comp = preprocess_data(data=row["trace"][:, :branch_point + row["trace"].shape[1] // 10])
    comp = preprocess_data(data=trace)
    try:
        info = compute_sim_info(data=comp)
    except networkx.exception.NetworkXError:
        raise
    for measure in MEASURES:
        # np.save(os.path.join("output_long", measure.lower(), str(s)), info[measure])
        np.save(os.path.join("output_train", measure.lower(), file.replace(".mat", "")), info[measure])
    print(file)


def compute_all_info(n_workers):
    # old_data = load_all_mat_data(
    #     path="GARD-model/Matlab/SourceCode/Doron_Lancet_GARD_Next_generation/GARD_v10/data")
    # data = load_all_mat_data_branching(
    #     path="GARD-model/Matlab/SourceCode/Doron_Lancet_GARD_Next_generation/GARD_v10/new_causality")
    # rows = [(idx, row, old_data[row["mother.id"]]) for idx, row in data.iterrows()]
    # files = os.listdir("GARD-model/Matlab/SourceCode/Doron_Lancet_GARD_Next_generation/GARD_v10/data_long")
    files = os.listdir("optim/traces")
    with Pool(n_workers) as pool:
        pool.map(compute_long_info, files)


if __name__ == "__main__":
    args = parse_args()
    set_seed(s=args.seed)
    compute_all_info(n_workers=args.n_workers)
    exit()
    df = load_all_mat_data(attribute="trace")
    for seed, traj in enumerate(df):
        # print(seed)
        # if len(traj) < 5000:
        #     continue
        # if seed >= 59:
        #     continue
        # traj = traj[traj["is_parent"]]
        # for i, composomes in enumerate(traj):
            # composomes = np.stack(d["n"].to_numpy()).T
        composomes = preprocess_data(data=traj[:, :traj.shape[1] // 2])
        try:
            information = compute_sim_info(data=composomes)
        except networkx.exception.NetworkXError:
            raise

        for measure in MEASURES:
            np.save(os.path.join("output_mothers", measure.lower(), str(seed)), information[measure])

        print(seed)
