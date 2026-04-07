import os

import networkx.exception
import numpy as np
from scipy.stats import zscore

from information import mutual_information_matrix, minimum_information_bipartition, local_phi_id, local_phi_r, \
    preprocess_data
from plotting import load_data, load_mat_data, load_all_mat_data
from utils import set_seed, parse_args, MEASURES


def compute_sim_info(data):
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


if __name__ == "__main__":
    args = parse_args()
    set_seed(s=args.seed)

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
        composomes = preprocess_data(data=traj)
        try:
            information = compute_sim_info(data=composomes)
        except networkx.exception.NetworkXError:
            raise

        # for measure in MEASURES:
        #     np.save(os.path.join("output_fluxes", measure.lower(), str(seed)), information[measure])

        print(seed)
