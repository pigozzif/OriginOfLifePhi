import os
import time

import numpy as np
from matplotlib import pyplot as plt

from utils import set_seed, parse_args


def seed_population(ng, ntot, nmin):
    # Each molecule type is repeated nTOT times
    pool = np.repeat(np.arange(ng), ntot)
    # Randomly select Nmin molecules from the pool (without replacement)
    selected = np.random.choice(pool, size=nmin, replace=False)
    # Count how many molecules of each type were picked
    return np.bincount(selected, minlength=ng)


def gard_step(n, Nmax, kf, kb, rho, beta, dt):
    n_mol = n.sum()
    if n_mol >= Nmax or n_mol == 0:
        return n, False
    # kinetic flux for each molecule type
    frac = n / max(n_mol, 1)
    # stochastic update (Poisson sampling)
    join = np.random.poisson((kf * rho * n_mol) * (1 + beta.dot(frac)) * dt)
    leave = np.random.poisson((kb * n) * (1 + beta.dot(frac)) * dt)
    n = np.clip(n + join - leave, 0, None)
    return n, True


def gard_generation(
        n,
        beta,
        NG,
        ntot,
        Nmax,
        kf,
        kb,
        rho,
        dt=0.05,
        max_steps=1_000
):
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
        n, ok = gard_step(n=n, Nmax=Nmax, kf=kf, kb=kb, rho=rho, beta=beta, dt=dt)
        if not ok:
            break
        # Apply updates, allowing both positive and negative fluxes
        ns.append(n)

    # ----- fission -----
    daughter1 = np.random.binomial(n, 0.5)
    daughter2 = n - daughter1

    return daughter1 if np.random.rand() < 0.5 else daughter2, ns


def gard_multigenerational(
        generations,
        NG=100,
        ntot=1000,
        Nmax=80,
        kf=1e-3,
        kb=1e-5,
        A=-4,
        sigma=4
):
    rho = np.ones(NG) / NG

    # catalytic matrix
    # log10_beta = np.random.normal(loc=A, scale=sigma, size=(NG, NG))
    # beta = 10 ** log10_beta
    beta = np.random.lognormal(mean=A, sigma=sigma, size=(NG, NG))
    # print(np.percentile(np.log10(beta), [1, 50, 99, 99.9]))

    # plt.imshow(beta, cmap='magma')
    # plt.colorbar(label='log10 β')
    # vals = beta.flatten()
    # plt.hist(vals, bins=100)
    # plt.show()
    # exit()

    # initial seed assembly
    n = seed_population(ng=NG,
                        ntot=ntot,
                        nmin=Nmax // 2)
    print(np.sum(n))
    exit()

    for gen in range(generations):
        n, ns = gard_generation(n, beta, NG, ntot, Nmax, kf, kb, rho)
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
    # with open(file_name, "w") as file:
    #     file.write(";".join(["i", "step", "is_parent", "is_daughter", "elapsed.sec", "n"]) + "\n")

    s = time.time()
    for g, d, comps in gard_multigenerational(generations=args.n_gen,
                                              kf=args.kf,
                                              kb=args.kb,
                                              A=args.A,
                                              sigma=args.sigma):
        for i, comp in enumerate(comps):
            with open(file_name, "a") as file:
                file.write(";".join([str(g),
                                     str(i),
                                     str(i == (len(comps) - 1)),
                                     str(i == 0),
                                     str(time.time() - s),
                                     "/".join([str(c) for c in comp])]) + "\n")
