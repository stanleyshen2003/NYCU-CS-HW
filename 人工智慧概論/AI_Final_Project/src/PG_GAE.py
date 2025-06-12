import os
import gym
from itertools import count
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler
import buildEnv
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from buildEnv import state_preprocess

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

        
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)     
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 500
        self.double()

        # calculate the observation space size
        self.observation_dim = 1
        for i in env.observation_space.shape:
            self.observation_dim *= i

        self.shared_layer1 = nn.Linear(self.observation_dim, self.hidden_size)
        self.shared_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.shared_layer3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.action_layer = nn.Linear(self.hidden_size, self.action_dim)
        self.value_layer = nn.Linear(self.hidden_size, 1)
        
        # action & reward memory
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        x = self.shared_layer1(Tensor(state).reshape(self.observation_dim))
        x = F.relu(x)
        x = self.shared_layer2(x)
        x = F.relu(x)
        x = self.shared_layer3(x)
        x = F.sigmoid(x)
        action_prob = self.action_layer(x)
        state_value = self.value_layer(x)

        return action_prob, state_value


    def select_action(self, state):
        state = state.reshape(-1)
        tempstate = state        
        for i in range(12):
            for j in range(4):
                tempstate[i*4+j] = (state[44+j] - state[i*4+j])/state[44+j]
        state = torch.Tensor(tempstate)
        if torch.cuda.is_available():
            state = state.cuda()
        action, state_value= self.forward(state)
        m = Categorical(logits=action)
        action = m.sample()
        
        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()


    def calculate_loss(self, gamma=0.999, lambda_ = 0.99):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] 
        value_losses = [] 
        returns = []

        discounted_sum = 0
        for reward in reversed(self.rewards):
            discounted_sum = reward + gamma * discounted_sum
            returns.append(discounted_sum)
        returns.reverse()
        returns = torch.Tensor(returns)

        if torch.cuda.is_available():
            returns = returns.cuda()

        returns = (returns - returns.mean()) / (returns.std())
        returns = returns.detach()

        log_probs = [action.log_prob for action in saved_actions]
        values = [action.value for action  in saved_actions]

        advantages = GAE(gamma, lambda_, None)(self.rewards, values)
        advantages = advantages.detach()

        action_log_probs = torch.stack(log_probs, dim=0)
        values = torch.stack(values, dim=0)[:,0]

        policy_losses = -(advantages * action_log_probs).sum()
        value_losses = F.mse_loss(values, returns).sum()
        loss = policy_losses + value_losses
        
        return loss

    def clear_memory(self):
        del self.rewards[:]
        del self.saved_actions[:]

class GAE:
    def __init__(self, gamma, lambda_, num_steps):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_steps = num_steps          # set num_steps = None to adapt full batch

    def __call__(self, rewards, values, done=False):
        advantages = []
        advantage = 0
        next_value = 0
        t = 0
        for r, v in zip(reversed(rewards), reversed(values)):
            t += 1
            td_error = r + next_value * self.gamma - v
            advantage = td_error + advantage * self.gamma * self.lambda_
            next_value = v
            advantages.insert(0, advantage)
            if self.num_steps is not None and t > self.num_steps:
                break
        advantages = torch.Tensor(advantages)
        if torch.cuda.is_available():
            advantages = advantages.cuda()
        advantages = (advantages - advantages.mean()) / (advantages.std())
        return advantages

def train(lr=0.001, lr_decay=0.999, gamma=0.999, lambda_ = 0.999):
    random_seed = 10
    global env 
    env = buildEnv.createEnv(2330)
    env.seed(random_seed)  
    torch.manual_seed(random_seed)  

    writer = SummaryWriter("./tb_record_1/Policy_Gradient_GAE-{}-{}-{}-{}".format(lr, lr_decay, gamma, lambda_))
    
    model = Policy()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # model.load_state_dict(torch.load('./Tables/PG_GAE-0.001-0.999(10000).pth'))
    
    #scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=lr_decay)
    
    ewma_reward = 0

    pre_reward = 0
    same_count = 0
    
    for i_episode in range(10000):
        state = env.reset()
        ep_reward = 0
        t = 0

        sell = 0
        buy = 0
        while True:
            t += 1
            action = model.select_action(state)
            if action == 0: sell += 1
            else: buy += 1

            state, reward, done, _ = env.step(action)
            ep_reward += reward
            model.rewards.append(reward)

            if done:
                break
        
        optimizer.zero_grad()
        loss = model.calculate_loss(gamma, lambda_)
        loss.backward()
        optimizer.step()
        #scheduler.step()
        model.clear_memory()
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\treward: {}\tprofit: {}\tloss: {}\tsell: {}\tbuy: {}'.format(i_episode, ep_reward, env._total_profit, loss, sell, buy))

        writer.add_scalar('EWMA reward', ewma_reward, i_episode)
        writer.add_scalar('Episode reward', ep_reward, i_episode)
        writer.add_scalar('Episode profit', env._total_profit, i_episode)
        writer.add_scalar('Episode loss', loss, i_episode)
        """
        if ep_reward == pre_reward:
            same_count += 1
        else:
            same_count = 0
        pre_reward = ep_reward
        if same_count >= 500:
            pre_reward = 0
            same_count = 0
            break
        """
        if (i_episode - 1) % 100 == 0:
            torch.save(model.state_dict(), './Tables/PG_GAE-{}-{}.pth'.format(lr, lr_decay))
    torch.save(model.state_dict(), './Tables/PG_GAE-{}-{}.pth'.format(lr, lr_decay))
    env.close()
    return -ewma_reward
   

def test(n_episodes=10):
    w = SummaryWriter('tb_record_1/comp_profit_train/REINFORCE')
    env = buildEnv.createEnv(2330, frame_bounds=(12, 1000))
    model = Policy()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    model.load_state_dict(torch.load('./Tables/PG_GAE.pth'))
    
    render = True
    max_episode_len = 10000
    state = env.reset()
    for t in range(max_episode_len+1):
        action = model.select_action(state)
        state, reward, done, info = env.step(action)
        w.add_scalar('Profit', env._total_profit, t)
        if done:
            break
    print('Reward: {}\tProfit: {}'.format(env._total_reward, env._total_profit))
    env.close()
    

if __name__ == '__main__':
    #lr = 0.001
    env = buildEnv.createEnv(2330)
    
    #env.seed(random_seed)  
    #torch.manual_seed(random_seed)  
    #train()
    test()