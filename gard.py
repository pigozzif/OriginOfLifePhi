import os
import time

import numpy as np
import pandas as pd

from utils import set_seed, parse_args


def seed_population(ng, ntot, nmin):
    # Construct the environmental pool
    # Each molecule type is repeated nTOT times
    pool = np.repeat(np.arange(ng), ntot)
    # Randomly select Nmin molecules from the pool (without replacement)
    selected = np.random.choice(pool, size=nmin, replace=False)
    # Count how many molecules of each type were picked
    return np.bincount(selected, minlength=ng)


def gard_generation(
        n,
        beta,
        NG,
        Nmax,
        kf,
        kb,
        rho,
        dt=0.05,
        max_steps=10_000
):
    # """Run one growth cycle until Nmax, then fission into Nmax/2 daughters."""
    # while n.sum() < Nmax:
    #     n_molecules = n.sum()
    #     second_terms = np.array([(1 + np.sum(beta[i, :] * n / max(n_molecules, 1))) for i in range(NG)])
    # calculate propensities
    #     join_rates = np.array([
    #         (kf * rho[i] * n_molecules) * second_terms[i] for i in range(NG)
    #     ])
    #     leave_rates = np.array([
    #         (kb * n[i]) * second_terms[i] for i in range(NG)
    #     ])

    #     a = np.concatenate([join_rates, leave_rates])
    #     a0 = np.sum(a)
    #     if a0 <= 0:
    #         break

    # Gillespie steps
    #     r1, r2 = np.random.uniform(), np.random.uniform()
    #     tau = (1.0 / a0) * np.log(1.0 / r1)

    #     reaction_index = np.searchsorted(np.cumsum(a), r2 * a0)
    #     if reaction_index < NG:
    #         n[reaction_index] += 1
    #     else:
    #         j = reaction_index - NG
    #         if n[j] > 0:
    #             n[j] -= 1

    # fission: random split into 2 daughters of Nmax/2 each
    # daughter = np.zeros_like(n)
    # indices = np.repeat(np.arange(NG), n)  # expanded composition
    # np.random.shuffle(indices)
    # selected = indices[:Nmax // 2]  # one daughter chosen
    # for idx in selected:
    #     daughter[idx] += 1

    # return daughter
    """
        One GARD growth cycle using Poisson-process dynamics
        (after Segrè et al. 2000, Eq. 4), until reaching Nmax molecules,
        then fissioning into two daughters of size Nmax/2.

        Parameters
        ----------
        n : ndarray
            Initial molecule counts (shape NG)
        beta : ndarray
            Catalytic matrix (shape NG×NG)
        NG : int
            Number of molecular species
        Nmax : int
            Maximum assembly size before fission
        kf, kb : float
            Joining / leaving rate constants
        rho : ndarray
            Environmental abundances (shape NG)
        dt : float
            Poisson time step
        max_steps : int
            Safety cap on iterations
        """
    ns = [n]
    for _ in range(max_steps):
        n_mol = n.sum()
        if n_mol >= Nmax or n_mol == 0:
            break
        # kinetic flux for each molecule type
        flux = (kf * rho * n_mol - kb * n) * (1.0 + (beta @ n) / max(n_mol, 1))
        # stochastic update (Poisson sampling)
        dn = np.random.poisson(flux * dt)
        # Apply updates, allowing both positive and negative fluxes
        n = np.clip(n + dn, 0, None)
        ns.append(n)

    # ----- fission -----
    daughter = np.random.binomial(n, 0.5)

    return daughter, ns


def gard_multigenerational(
        generations=10,
        NG=100,
        Nmax=80,
        kf=1e-2,
        kb=1e-5,
        A=-4.0,
        sigma=4.0
):
    rho = np.ones(NG) / NG

    # catalytic matrix
    beta = np.random.lognormal(mean=A, sigma=sigma, size=(NG, NG))

    # initial seed assembly
    # n = np.zeros(NG, dtype=int)
    # n[np.random.randint(0, NG)] = 1
    n = seed_population(ng=NG,
                        ntot=1000,
                        nmin=Nmax // 2)

    for gen in range(generations):
        n, ns = gard_generation(n, beta, NG, Nmax, kf, kb, rho)
        yield gen, n.copy(), ns


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    file_name = os.path.join("output", ".".join([str(args.seed), "txt"]))
    # if os.path.exists(file_name):
    #     temp_data = pd.read_csv(file_name)
    #     if len(temp_data) >= args.n_gen:
    #         print(file_name)
    #         exit()
    with open(file_name, "w") as file:
        file.write(";".join(["i", "step", "is_parent", "is_daughter", "elapsed.sec", "n"]) + "\n")

    s = time.time()
    for g, d, comps in gard_multigenerational(generations=args.n_gen):
        for i, comp in enumerate(comps):
            with open(file_name, "a") as file:
                file.write(";".join([str(g),
                                     str(i),
                                     str(i == (len(comps) - 1)),
                                     str(i == 0),
                                     str(time.time() - s),
                                     "/".join([str(c) for c in comp])]) + "\n")
