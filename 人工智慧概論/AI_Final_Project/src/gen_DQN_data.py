from DDQN import gen_offline_data
from buildEnv import createEnv
import numpy as np
import pickle

episodes = 10
stock = 2330
env = createEnv(2330)

data = gen_offline_data(episodes, env)

with open('Trajectory_Transformer/trajectory/datasets/Medium/DDQN_{}_{}'.format(episodes, stock) + '.pkl', 'wb') as f:
    pickle.dump(data, f)
