import copy
import os

import gymnasium
import numpy as np
from gymnasium.vector.utils import spaces

from information import preprocess_data
from main import compute_sim_info
from plotting import load_all_mat_data
from utils import parse_args


class GardEnv(gymnasium.Env):
    def __init__(self,
                 generations,
                 seed,
                 p,
                 NG=100,
                 ntot=1000,
                 Nmax=80,
                 kf=1e-3,
                 kb=1e-5,
                 A=-4,
                 sigma=4,
                 max_steps=1000):
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
        self.beta = np.random.lognormal(mean=A, sigma=sigma, size=(NG, NG))
        # initial seed assembly
        self.seed_data = load_all_mat_data(attribute="history")[seed]
        self.seed_data = self.seed_data[:, :int(self.seed_data.shape[1] * p) + 2]
        self.ns = [n for n in self.seed_data]
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
        self.ns = [n for n in self.seed_data]
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
        data = preprocess_data(data=np.array(self.ns).T)
        info = compute_sim_info(data=data)
        reward = np.nanmedian(info["emergence"])
        self.i += 1
        n_mol = self.n.sum()
        if n_mol >= self.Nmax or n_mol == 0 or self.i >= self.max_steps:
            daughter1 = np.random.binomial(self.n, 0.5)
            daughter2 = self.n - daughter1
            self.n = daughter1 if np.random.rand() < 0.5 else daughter2
            self.i = 0
            self.t += 1
        print(self.t)
        return self._get_obs(), reward, self.t >= self.gen, False, {}


class GreedyAgent(object):
    def __init__(self, n_action_samples=1):
        self.n_action_samples = n_action_samples

    def valid_actions(self, env):
        actions = []
        for idx in range(env.NG):
            add_action = 2 * idx
            remove_action = 2 * idx + 1

            # add is always allowed
            actions.append(add_action)

            # remove only if there is something to remove
            if env.n[idx] > 0:
                actions.append(remove_action)

        return actions

    def evaluate_action(self, env, action):
        # one-step lookahead on a copy
        env_copy = copy.deepcopy(env)
        _, reward, terminated, truncated, _ = env_copy.step(action)
        return reward

    def act(self, env):
        actions = self.valid_actions(env)

        best_action = None
        best_reward = -np.inf

        for action in actions:
            reward = self.evaluate_action(env, action)
            if reward > best_reward:
                best_reward = reward
                best_action = action

        return best_action, best_reward


class BeamSearchAgent:
    def __init__(self, beam_width=5, depth=2, stochastic_samples=3, discount=0.95):
        self.beam_width = beam_width
        self.depth = depth
        self.stochastic_samples = stochastic_samples
        self.discount = discount

    def evaluate_transition(self, env, action):
        """
        Apply one action from the current env state and return:
        - mean reward over stochastic rollouts
        - resulting state snapshot
        - whether episode ended
        """
        root_state = env.get_state()
        rewards = []
        next_states = []
        dones = []

        for _ in range(self.stochastic_samples):
            env.set_state(root_state)
            _, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            next_states.append(env.get_state())
            dones.append(terminated or truncated)

        env.set_state(root_state)

        mean_reward = float(np.mean(rewards))
        # pick one representative next state
        rep_state = next_states[int(np.argmax(rewards))]
        done = bool(np.any(dones))
        return mean_reward, rep_state, done

    def act(self, env):
        initial_state = env.get_state()

        # Each beam item is:
        # {
        #   "state": saved env state,
        #   "actions": list of actions taken,
        #   "score": cumulative discounted reward,
        #   "done": bool
        # }
        beam = [{
            "state": initial_state,
            "actions": [],
            "score": 0.0,
            "done": False,
        }]

        for d in range(self.depth):
            candidates = []

            for item in beam:
                if item["done"]:
                    candidates.append(item)
                    continue

                env.set_state(item["state"])
                actions = env.valid_actions()

                for action in actions:
                    reward, next_state, done = self.evaluate_transition(env, action)
                    score = item["score"] + (self.discount ** d) * reward

                    candidates.append({
                        "state": next_state,
                        "actions": item["actions"] + [action],
                        "score": score,
                        "done": done,
                    })

            if not candidates:
                break

            candidates.sort(key=lambda x: x["score"], reverse=True)
            beam = candidates[:self.beam_width]

        env.set_state(initial_state)

        best = max(beam, key=lambda x: x["score"])
        first_action = best["actions"][0] if best["actions"] else env.valid_actions()[0]
        return first_action, best


if __name__ == "__main__":
    args = parse_args()
    env = GardEnv(generations=100,
                  seed=args.seed,
                  p=args.p)

    # model = DQN(
    #     "MlpPolicy",
    #     env,
    #     verbose=1
    # )

    # model.learn(total_timesteps=1_000_000)
    # exit()
    agent = BeamSearchAgent()
    file_name = os.path.join("causality", f"{args.seed}.txt")
    with open(file_name) as file:
        file.write(";".join(["gen", "step", "n_sum", "phi", "comp"]) + "\n")

    obs, info = env.reset()

    done = False
    trajectory = []

    while not done:
        action, predicted_reward = agent.act(env)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(env.i, env.t)
        # trajectory.append({
        #     "action": action,
        #     "reward": reward,
        #     "predicted_reward": predicted_reward,
        #     "n_sum": env.n.sum(),
        #     "generation": env.t,
        # })
        # trajectory.append({
        #     "action": action,
        #     "reward": reward,
        #     "planned_score": plan["score"],
        #     "planned_actions": plan["actions"],
        #     "generation": env.t,
        #     "n_total": env.n.sum(),
        # })
        with open(file_name) as file:
            file.write(";".join([str(env.t), str(env.i), str(env.n.sum()), str(reward),
                                 "/".join([str(e) for e in env.n])]) + "\n")
