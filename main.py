import networkx.exception
import numpy as np

from information import mutual_information_matrix, minimum_information_bipartition, local_phi_id, local_phi_r
from plotting import load_data
from utils import set_seed, parse_args, MEASURES


def preprocess_data(data):
    gmean = np.exp(np.mean(np.log(data + 1e-12), axis=0, keepdims=True))
    data = np.log((data + 1e-12) / gmean)
    return data[:-1, :]


def compute_sim_info(data):
    data = data.astype(np.float64, copy=False)
    info = {}
    mi_mat = mutual_information_matrix(data, alpha=1, bonferonni=False, lag=1)
    mib = minimum_information_bipartition(mi_mat, noise=False)
    component_1 = data[mib[0], :].mean(axis=0)
    component_2 = data[mib[1], :].mean(axis=0)
    data_reduced = np.vstack((component_1, component_2))
    phi_results = local_phi_id(0, 1, data_reduced)
    info["synergy"] = phi_results.nodes[(((0, 1),), ((0, 1),))]["pi"]
    info["causation"] = phi_results.nodes[(((0, 1),), ((0,),))]["pi"] + phi_results.nodes[(((0, 1),), ((1,),))]["pi"]
    info["redundancy"] = phi_results.nodes[(((0,), (1,)), ((0,), (1,)))]["pi"] + \
                         phi_results.nodes[(((0,), (1,)), ((0,), (1,)))]["pi"]
    info["integrated"] = local_phi_r(phi_results)
    info["emergence"] = info["synergy"] + info["causation"]
    return info


if __name__ == "__main__":
    args = parse_args()
    set_seed(s=args.seed)
    with open("info.txt", "w") as file:
        file.write(";".join(["seed"] + MEASURES) + "\n")

    df = load_data()
    for (seed,), traj in df.groupby(["seed"]):
        print(seed)
        # if len(traj) < 5000:
        #     continue
        # if seed >= 59:
        #     continue
        traj = traj[traj["is_parent"]]
        composomes = np.stack(traj["n"].to_numpy()).T
        composomes = preprocess_data(data=composomes)
        try:
            information = compute_sim_info(data=composomes)
        except networkx.exception.NetworkXError:
            continue

        vals = [str(seed)]
        for measure in MEASURES:
            vals.append("/".join([str(round(v, 4)) for v in information[measure]]))
        with open("info.txt", "a") as file:
            file.write(";".join(vals) + "\n")
        print(seed)
