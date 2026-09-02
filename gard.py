import os
import random
import time

import numpy as np
from matplotlib import pyplot as plt

from utils import parse_args, set_seed


def seed_population(ng, ntot, nmin):
    # Each molecule type is repeated nTOT times
    pool = np.repeat(np.arange(ng), ntot)
    # Randomly select Nmin molecules from the pool (without replacement)
    selected = np.random.choice(pool, size=nmin, replace=False)
    # Count how many molecules of each type were picked
    return np.bincount(selected, minlength=ng)


def gard_step(n, Nmax, kf, kb, rho, beta, dt):
    # n_mol = n.sum()
    # if n_mol >= Nmax or n_mol == 0:
    #     return n, False
    # kinetic flux for each molecule type
    # frac = n / max(n_mol, 1)
    # stochastic update (Poisson sampling)
    # join = np.random.poisson((kf * rho * n_mol) * (1 + beta.dot(frac)) * dt)
    # leave = np.random.poisson((kb * n) * (1 + beta.dot(frac)) * dt)
    # n = np.clip(n + join - leave, 0, None)
    # return n, True
    n = n.copy()
    n_mol = n.sum()

    if n_mol >= Nmax or n_mol == 0:
        return n, False

    bn = 1 + (beta @ n) / n_mol

    join_rates = kf * rho * n_mol * bn
    leave_rates = kb * n * bn

    rates = np.concatenate([join_rates, leave_rates])
    rates = np.maximum(rates, 0)

    total = rates.sum()
    if total <= 0:
        return n, False

    mu = np.random.choice(len(rates), p=rates / total)

    NG = len(n)

    if mu < NG:
        n[mu] += 1
    else:
        i = mu - NG
        if n[i] > 0:
            n[i] -= 1
        else:
            return n, True

    return n, True


def gard_generation(
        n,
        beta,
        Nmax,
        kf,
        kb,
        rho,
        dt=0.05,
        max_steps=100
):
    """
    One GARD growth-fission generation using elementary stochastic events.

    Each molecular step changes the assembly by exactly one molecule:
    either +1 or -1 for one molecular species. The assembly grows until it
    reaches ``Nmax`` molecules, dies out at size 0, or hits ``max_steps``.
    Fission is applied only if the assembly reached ``Nmax``.

    ``dt`` is accepted for API compatibility but is not used by the
    event-based update.
    """
    n = n.copy().astype(int)
    ns = [n.copy()]

    reached_fission_size = False

    for _ in range(max_steps):
        n_mol = n.sum()

        if n_mol >= Nmax:
            reached_fission_size = True
            break

        if n_mol == 0:
            break

        n, moved = gard_step(n, Nmax, kf, kb, rho, beta, dt=dt)

        if not moved:
            break

        ns.append(n.copy())

    # If the assembly died out or stalled before reaching Nmax, do not fission.
    if not reached_fission_size and n.sum() < Nmax:
        return n, ns

    # ----- fission -----
    # Randomly partition the parent into two daughters and continue with one.
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
    """Yield successive GARD generations.

    The catalytic matrix follows the common GARD convention
    ``log10(beta_ij) ~ Normal(A, sigma)``. Each yielded item is
    ``(generation_index, daughter_for_next_generation, molecular_history)``.
    """
    if generations < 0:
        raise ValueError("generations must be non-negative")

    rho = np.ones(NG, dtype=float) / NG

    log10_beta = np.random.normal(loc=A, scale=sigma, size=(NG, NG))
    beta = 10.0 ** log10_beta

    n = seed_population(ng=NG, ntot=ntot, nmin=Nmax // 2)

    for gen in range(generations):
        n, history = gard_generation(n, beta, Nmax, kf, kb, rho)
        yield gen, n.copy(), history


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    file_name = os.path.join("ami_data/traces", ".".join([str(args.seed), "txt"]))
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
