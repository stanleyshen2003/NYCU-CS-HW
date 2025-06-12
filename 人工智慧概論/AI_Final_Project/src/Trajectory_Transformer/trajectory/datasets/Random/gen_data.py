import gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions 
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import sys
from buildEnv import createEnv, state_preprocess

quat_type = 2330
env = createEnv(2330)
# env = gym.make('stocks-v0', frame_bound=(50, 100), window_size=10)

action_dim = env.action_space.n

episode = 50
T = 0
episode_data = {
    'observations':[],
    'next_observations':[],
    'actions':[],
    'rewards':[],
    'terminals':[]
}
for e in range(episode):
    env.seed(e)
    observation = env.reset()
    observation = state_preprocess(observation)
    while True:
        probs = np.random.rand(action_dim)
        action = np.random.choice(2, p=probs/np.sum(probs))
        next_observation, reward, done, _ = env.step(action)
        next_observation = state_preprocess(next_observation)
        episode_data['observations'].append(observation.reshape(-1).astype('float32'))
        episode_data['next_observations'].append(next_observation.reshape(-1).astype('float32'))
        episode_data['actions'].append(probs)
        episode_data['rewards'].append(np.array([reward]).astype('float32'))
        episode_data['terminals'].append(done)
        if done:
            break

for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
    episode_data[k] = np.stack(episode_data[k])
with open('Trajectory_Transformer/trajectory/datasets/Random/TT_stock_{}_{}'.format(quat_type, episode) + '.pkl', 'wb') as f:
    pickle.dump(episode_data, f)