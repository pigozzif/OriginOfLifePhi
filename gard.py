import os
import time

import numpy as np

from utils import set_seed, parse_args


def gard_generation(
        n,
        beta,
        NG,
        Nmax,
        kf,
        kb,
        rho
):
    """Run one growth cycle until Nmax, then fission into Nmax/2 daughters."""
    while n.sum() < Nmax:
        n_molecules = n.sum()
        second_terms = np.array([(1 + np.sum(beta[i, :] * n / max(n_molecules, 1))) for i in range(NG)])
        # calculate propensities
        join_rates = np.array([
            (kf * rho[i] * n_molecules) * second_terms[i] for i in range(NG)
        ])
        leave_rates = np.array([
            (kb * n[i]) * second_terms[i] for i in range(NG)
        ])

        a = np.concatenate([join_rates, leave_rates])
        a0 = np.sum(a)
        if a0 <= 0:
            break

        # Gillespie steps
        r1, r2 = np.random.uniform(), np.random.uniform()
        tau = (1.0 / a0) * np.log(1.0 / r1)

        reaction_index = np.searchsorted(np.cumsum(a), r2 * a0)
        if reaction_index < NG:
            n[reaction_index] += 1
        else:
            j = reaction_index - NG
            if n[j] > 0:
                n[j] -= 1

    # fission: random split into 2 daughters of Nmax/2 each
    daughter = np.zeros_like(n)
    indices = np.repeat(np.arange(NG), n)  # expanded composition
    np.random.shuffle(indices)
    selected = indices[:Nmax // 2]  # one daughter chosen
    for idx in selected:
        daughter[idx] += 1

    return daughter


def gard_multigenerational(
        generations=10,
        NG=100,
        Nmax=100,
        kf=1e-2,
        kb=1e-4,
        A=-4.0,
        sigma=4.0
):
    rho = np.ones(NG) / NG

    # catalytic matrix
    beta = np.random.lognormal(mean=A, sigma=sigma, size=(NG, NG))

    # initial seed assembly
    n = np.zeros(NG, dtype=int)
    n[np.random.randint(0, NG)] = 1

    for gen in range(generations):
        n = gard_generation(n, beta, NG, Nmax, kf, kb, rho)
        yield gen, n.copy()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    file_name = os.path.join("output", ".".join([str(args.seed), "txt"]))
    with open(file_name, "w") as file:
        file.write(";".join(["i", "elapsed.sec", "n"]) + "\n")

    s = time.time()
    for g, comp in gard_multigenerational(generations=args.n_gen):
        with open(file_name, "a") as file:
            file.write(";".join([str(g),
                                 str(time.time() - s),
                                 "/".join([str(c) for c in comp])]) + "\n")
