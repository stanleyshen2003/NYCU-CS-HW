import buildEnv
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter






def test(env):
    """
    Test on the given environment.
    average50 : the list of last 50 day's average
    average200 : the list of last 200 day's average
    """
    
    average50 = list()                  
    average200 = list()
    for i in range(env.frame_bound[1]):
        if(i>=50):
            average50.append(np.sum(env.signal_features[i-50:i,0:4]) / 50/4)
        if(i>=200):
            average200.append(np.sum(env.signal_features[i-200:i, 0:4]) / 200/4)
    count = 0                                   # count the number of iteration
    env.reset()                                 
    count1 = 0                                  # count the action buy
    count0 = 0                                  # count the action sell
    w = SummaryWriter('tb_record_1/comp_profit_train/baseline')
    t = 0
    while True:
        if count<200:                           # skip to 201 th day
            env.step(0)
            count = count+1
            w.add_scalar('Profit', env._total_profit, t)
            t+=1
            continue
        # use the Moving Average Crossover method to decide whether to buy or not
        if average50[count-200+150] > average200[count-200]:
            action = 1
            count1 += 1
        else:
            action = 0
            count0 += 1

        next_state, _, done, _ = env.step(action)
        w.add_scalar('Profit', env._total_profit, t)
        t+=1
        
        count = count + 1
        if done:
            break

    print("count action - sell : " + str(count1))
    print("count action - sell : " + str(count0))
    print("total profit : " + str(env._total_profit))




if __name__ == "__main__":
    env = buildEnv.createEnv(2330, frame_bounds=(12,1000))        
    test(env)