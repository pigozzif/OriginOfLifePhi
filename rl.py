import os
import time
from multiprocessing import Pool

import gymnasium
import numpy as np
from gymnasium.vector.utils import spaces
from scipy.io import savemat, loadmat

from gard import gard_generation
from information import preprocess_data
from main import compute_sim_info
from utils import parse_args, set_seed, NMAX


class GardEnv(gymnasium.Env):
    def __init__(self,
                 generations,
                 seed,
                 p,
                 NG=100,
                 ntot=1000,
                 kf=1e-3,
                 kb=1e-5,
                 A=-4,
                 sigma=4,
                 Nmax=NMAX,
                 max_steps=100):
        super().__init__()
        self.gen = generations
        self.NG = NG
        self.ntot = ntot
        self.Nmax = Nmax
        self.kf = kf
        self.kb = kb
        self.A = A
        self.sigma = sigma
        self.max_steps = max_steps
        # Replace obs_dim with your actual state dimension
        self.rho = np.ones(NG) / NG
        # catalytic matrix
        # self.beta = np.random.lognormal(mean=A, sigma=sigma, size=(NG, NG))
        # initial seed assembly
        seed_data = loadmat(
            os.path.join("GARD-model", "Matlab", "SourceCode", "Doron_Lancet_GARD_Next_generation", "GARD_v10", "data",
                         f"GARD_run_seed_{seed + 1:03d}"))["o"]
        # catalytic matrix
        self.beta = seed_data["Beta"][0, 0]
        # initial seed assembly
        # self.seed_data = load_all_mat_data(attribute="history")[seed]
        self.seed_data = seed_data["history"][0, 0]
        self.seed_data = self.seed_data[:, :int(self.seed_data.shape[1] * p) + 3]
        self.ns = [self.seed_data[:, i] for i in range(self.seed_data.shape[1])]
        self.n = self.ns[-1]
        self.observation_space = spaces.Box(
            low=np.zeros_like(self.n), high=np.full_like(self.n, fill_value=np.inf), shape=self.n.shape,
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.NG * 2)
        self.i = 0
        self.t = 0

    def get_state(self):
        return {
            "n": self.n.copy(),
            "ns": [x.copy() for x in self.ns],
            "i": self.i,
            "t": self.t,
        }

    def set_state(self, state):
        self.n = state["n"].copy()
        self.ns = [x.copy() for x in state["ns"]]
        self.i = state["i"]
        self.t = state["t"]

    def valid_actions(self):
        actions = []
        for idx in range(self.NG):
            actions.append(2 * idx)  # add
            if self.n[idx] > 0:
                actions.append(2 * idx + 1)  # remove
        return actions

    def _get_obs(self):
        return self.ns[-1]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.i = 0
        self.t = 0
        self.ns.clear()
        self.ns = [self.seed_data[:, i] for i in range(self.seed_data.shape[1])]
        self.n = self.ns[-1]
        return self._get_obs(), {}

    def _decode_action(self, action):
        idx = action // 2
        if action % 2 == 0:
            self.n[idx] += 1
        else:
            if self.n[idx] > 0:
                self.n[idx] -= 1

    def step(self, action):
        # action update
        self._decode_action(action=action)
        # Apply updates, allowing both positive and negative fluxes
        self.ns.append(self.n.copy())
        data = preprocess_data(data=np.array(self.ns))
        info = compute_sim_info(data=data)
        reward = np.nanmedian(info["emergence"])
        # self.i += 1
        # n_mol = self.n.sum()
        # if n_mol >= self.Nmax or n_mol == 0 or self.i >= self.max_steps:
        #     daughter1 = np.random.binomial(self.n, 0.5)
        #     daughter2 = self.n - daughter1
        #     self.n = daughter1 if np.random.rand() < 0.5 else daughter2
        #     self.i = 0
        #     self.t += 1
        # print(self.t)
        n = self.ns.pop(-1)
        self.n = self.ns[-1].copy()
        return n, reward, self.t >= self.gen, False, {}

    def step_long(self, a=None):
        if a is not None:
            self._decode_action(action=a)
        n, ns = gard_generation(n=self.n,
                                beta=self.beta,
                                Nmax=self.Nmax,
                                kf=self.kf,
                                kb=self.kb,
                                rho=self.rho,
                                max_steps=self.max_steps)
        self.ns.extend(ns)
        self.ns.append(n.copy())
        self.n = n
        self.i += len(ns)
        self.t += 1


def get_reward(args):
    d1, a, beta, kf, kb, rho = args
    _, d2 = gard_generation(n=d1[:, -1],
                            beta=beta,
                            Nmax=NMAX,
                            kf=kf,
                            kb=kb,
                            rho=rho,
                            max_steps=10_000)
    d = np.hstack([d1, np.array(d2).T])
    data = preprocess_data(data=d)
    try:
        inf = compute_sim_info(data=data)
    except:
        inf = compute_sim_info(data=data)
        print(np.sum(np.isinf(data)), np.sum(np.isnan(data)))
        raise
    # print(a)
    return np.nanmedian(inf["emergence"]), a


def decode_action(env, a):
    idx = a // 2
    n = env.n.copy()
    if a % 2 == 0:
        n[idx] += 1
    else:
        if n[idx] > 0:
            n[idx] -= 1
    return np.array(env.ns + [n]).T


class GreedyAgent(object):
    def __init__(self, n_workers=1, n_action_samples=1):
        self.n_workers = n_workers
        self.n_action_samples = n_action_samples

    def valid_actions(self, env):
        actions = []
        for idx in range(env.NG):
            add_action = 2 * idx
            remove_action = 2 * idx + 1

            # add is always allowed
            actions.append(add_action)

            # remove only if there is something to remove
            if env.n[idx] > 0 and np.sum(env.n) > 1:
                actions.append(remove_action)

        return actions

    def evaluate_actions(self, env, actions):
        with Pool(self.n_workers) as pool:
            results = pool.map(get_reward,
                               [(decode_action(env=env, a=a).copy(), a, env.beta.copy(), env.kf, env.kb, env.rho) for a
                                in actions])
        return results

    def act(self, env, exp):
        actions = self.valid_actions(env)
        results = self.evaluate_actions(env=env, actions=actions)
        if exp == "max":
            best_idx = np.argmax([x[0] for x in results])
        else:
            best_idx = np.argmin([x[0] for x in results])
        return results[best_idx][1], results[best_idx][0]


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    env = GardEnv(generations=100,
                  seed=args.seed,
                  p=args.p,
                  kf=args.kf,
                  kb=args.kb,
                  Nmax=args.nmax)

    agent = GreedyAgent(n_workers=args.n_workers)
    file_name = os.path.join("causality", ".".join([str(args.seed), str(args.p).replace('.', ','),
                                                    str(args.kf).replace('.', ','), str(args.kb).replace('.', ','), str(args.nmax),
                                                    args.exp, "txt"]))
    with open(file_name, "w") as file:
        file.write(";".join(["gen", "step", "elapsed.sec", "n_sum", "phi"]) + "\n")

    obs, info = env.reset()
    start = time.time()

    while env.t < args.n_gen and env.n.sum() > 0:
        if args.exp == "base":
            predicted_reward = float(get_reward((np.array(env.ns).T, None, env.beta.copy(), env.kf, env.kb, env.rho))[0])
            env.step_long(a=None)
        else:
            action, predicted_reward = agent.act(env, exp=args.exp)
            env.step_long(a=action)
        with open(file_name, "a") as file:
            file.write(";".join([str(env.t), str(env.i), str(time.time() - start), str(env.n.sum()),
                                 str(predicted_reward)]) + "\n")
    savemat(
        file_name.replace("causality", os.path.join("causality", "traces")).replace(".txt", ".mat"),
        {"history":
            np.array(
                env.ns)})
