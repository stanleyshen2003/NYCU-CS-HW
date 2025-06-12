import numpy as np

class GaeSampleMemory(object):
    # one path for one agent
    class Path(object):
        def __init__(self):
            self.trajectories = []

        def __len__(self):
            sample_count = 0
            for t in self.trajectories:
                sample_count += len(t)
            return sample_count

        def append(self, sample):
            if len(self.trajectories) == 0 or self.trajectories[-1].transitions["done"][-1]:
                self.trajectories.append(GaeSampleMemory.Trajectory())
            self.trajectories[-1].append(sample) 

        def get_keys(self):
            return self.trajectories[0].get_keys()

        def get_observation_keys(self):
            return self.trajectories[0].get_observation_keys()

        def merge(self, key):
            results = [t.merge(key) for t in self.trajectories]
            return np.concatenate(results)

        def merge_observations(self, key):
            results = np.concatenate([t.merge_observations(key) for t in self.trajectories])
            return results

        def merge_next_observations(self, key):
            results = np.concatenate([t.merge_next_observations(key) for t in self.trajectories])
            return results

        def clear(self):
            self.trajectories = []

    class Trajectory(object):
        def __init__(self):
            self.transitions = {
                "observation": {},
                "action": [],
                "reward": [],
                "value": [],
                "logp_pi": [],
                "done": [],
            }

        def __len__(self):
            return len(self.transitions["action"])

        def append(self, sample):
            for key in sample:
                if key == "observation":
                    for obs_key in sample[key]:
                        if obs_key not in self.transitions[key]:
                            self.transitions[key][obs_key] = []
                        self.transitions[key][obs_key].append(sample[key][obs_key])
                else:
                    self.transitions[key].append(sample[key])

        def get_keys(self):
            return ["observation", "action", "reward", "value", "logp_pi", "done"]

        def get_observation_keys(self):
            return self.transitions["observation"].keys()

        def merge(self, key):
            merged_results = [s for s in self.transitions[key]]
            return merged_results

        def merge_observations(self, key):
            merged_results = [s for s in self.transitions["observation"][key]]
            return merged_results

        def merge_next_observations(self, key):
            merged_results = [s for s in self.transitions["observation"][key]]
            return merged_results[1:] + [merged_results[-1]]

    def __init__(self, config):
        self.config = config
        self.horizon = self.config["horizon"]
        self.use_return_as_advantage = self.config["use_return_as_advantage"]
        self.paths = []
        for _ in range(self.config["agent_count"]):
            self.paths.append(GaeSampleMemory.Path())
        self._sample_count = 0

    def __len__(self):
        sample_count = 0
        for index in range(self.config["agent_count"]):
            sample_count += len(self.paths[index])
        return sample_count

    def clear_buffer(self):
        if len(self) > 0:
            for index in range(self.config["agent_count"]):
                self.paths[index].clear()

    def append(self, index, sample):
        self.paths[index].append(sample)
        self._sample_count += 1

    def extract_batch(self, discount_gamma, discount_lambda, use_next_observation = False):
        returns = []
        advs = []

        for i in range(self.config["agent_count"]):
            _return, _adv = self.get_gae(i, discount_gamma, discount_lambda)
            returns.append(_return)
            advs.append(_adv)

        advs = np.concatenate(advs)
        advs = (advs - advs.mean()) / (advs.std() + 0.0001)
        batchs = {
            "return": np.concatenate(returns),
            "adv": advs,
        }
        for key in self.paths[0].get_keys():
            if key == "observation":
                batchs[key] = {}
                for obs_key in self.paths[0].get_observation_keys():
                    batchs[key][obs_key] = np.concatenate([self.paths[i].merge_observations(obs_key) for i in range(self.config["agent_count"])])
            else:
                # print(key)
                batchs[key] = np.concatenate([self.paths[i].merge(key) for i in range(self.config["agent_count"])])
        if use_next_observation:
            batchs["next_observation"] = {}
            for obs_key in self.paths[0].get_observation_keys():
                batchs["next_observation"][obs_key] = np.concatenate([self.paths[i].merge_next_observations(obs_key) for i in range(self.config["agent_count"])])

        return batchs

    def get_gae(self, index, discount_gamma, discount_lambda):
        returns, advs = [], []
        for trajectory in self.paths[index].trajectories:
            rewards = trajectory.merge("reward")
            values = trajectory.merge("value")
            dones = trajectory.merge("done")
            sample_count = len(rewards)
            if trajectory.transitions["done"][-1]:
                values.append(0)
            else:
                values.append(values[-1])
            for start in range(0, sample_count, self.horizon):
                _return, _adv = self._compute_gae(
                    rewards=rewards[start:start + self.horizon], 
                    values=values[start:start + self.horizon + 1], 
                    dones=dones[start:start + self.horizon], 
                    discount_gamma=discount_gamma, 
                    discount_lambda=discount_lambda)
                returns.append(_return)
                advs.append(_adv)

        return np.concatenate(returns), np.concatenate(advs)

    def _compute_discounted_return(self, rewards, dones, discount_gamma, discount_lambda):
        """Discounted Return Calculation 
        Args:
            rewards: A list of 1-d np.array, reward at every time step
            discount factor gamma will be automatically used
        
        Return:
            q_path: Q-value
        """
        q = 0
        q_path = []
        count = len(rewards) - 1
        for i in range(len(rewards)):
            if dones[count - i]:
                q = rewards[count - i]
            else:
                q = rewards[count - i] + discount_gamma * discount_lambda * q
            q_path.append(q)
        q_path.reverse()
        q_path = np.asarray(q_path)
        return q_path

    def _compute_gae(self, rewards, values, dones, discount_gamma, discount_lambda):
        """Generalized Advantage Estimation

        Args:
        rews: A list or 1-d np.array, reward at every time step
        values: A list or 1-d np.array, value estimation at every time step (include last value)
        discount factor gamma will be automatically used
        gae discount factor lambda will be automatically used

        Return:
            returns: discounted return
            adv: advantage estimation
        """
        delta = np.zeros_like(rewards, dtype=np.float32)
        for t in range(len(rewards)):
            if dones[t]:
                delta[t] = rewards[t] - values[t]
            else:
                delta[t] = rewards[t] + discount_gamma * values[t + 1] - values[t]
        adv = self._compute_discounted_return(delta, dones, discount_gamma, discount_lambda)
        # print(adv)
        # print(values[:len(adv)])
        returns = np.asarray(adv) + np.asarray(values[:len(adv)])

        if self.use_return_as_advantage:
            adv = returns

        return returns, adv
