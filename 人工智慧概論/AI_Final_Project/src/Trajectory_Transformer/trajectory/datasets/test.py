import numpy as np

import collections
import pickle
import numpy as np

with open('/home/kjlin0508/Course_work/AI_Intro/RL_for_Quatitatitive_Trading/Trajectory_Transformer/trajectory/datasets/Random/forex-v0.pkl', 'rb') as f:
    dataset = pickle.load(f)
print(dataset['observations'].shape)
print(dataset['next_observations'].shape)
print(dataset['actions'].shape)
print(dataset['rewards'].shape)
print(dataset['terminals'].shape)
#X = sobol_seq.i4_sobol_generate(2, 1000, np.random.randint(0, 100))
#print(X)
#view = k_means(X, n_clusters= 100)[0]
#view_point = []
#for i in range(len(view)):
#    view_point.append(np.argmin(np.linalg.norm(view[i]-X, axis=1)))
#print(view_point)
#with open('/home/kjlin0508/multi_objective_bo/trajectory-transformer/env/Functions/domain/domain_DRZ_1000.npy', 'rb') as f:
#    dataset = np.load(f)
#print(dataset)#