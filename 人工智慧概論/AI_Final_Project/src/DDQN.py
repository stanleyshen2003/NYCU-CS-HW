import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import random
from collections import deque
from torch import Tensor
import os
from tqdm import tqdm
import buildEnv
import math
import math
from torch.utils.tensorboard import SummaryWriter
total_rewards = []


class replay_buffer():
    '''
    A deque storing trajectories
    '''
    def __init__(self, capacity):
        self.capacity = capacity  # the size of the replay buffer
        self.memory = deque(maxlen=capacity)  # replay buffer itself

    def insert(self, state, action, reward, next_state, done):
        '''
        Insert a sequence of data gotten by the agent into the replay buffer.
        Parameter:
            state: the current state
            action: the action done by the agent
            reward: the reward agent got
            next_state: the next state
            done: the status showing whether the episode finish        
        Return:
            None
        '''
        
        self.memory.append([state, action, reward, next_state.reshape(48), done])

    def sample(self, batch_size):
        '''
        Sample a batch size of data from the replay buffer.
        Parameter:
            batch_size: the number of samples which will be propagated through the neural network
        Returns:
            observations: a batch size of states stored in the replay buffer
            actions: a batch size of actions stored in the replay buffer
            rewards: a batch size of rewards stored in the replay buffer
            next_observations: a batch size of "next_state"s stored in the replay buffer
            done: a batch size of done stored in the replay buffer
        '''
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, done = zip(*batch)
        return observations, actions, rewards, next_observations, done

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    '''
    The structure of the Neural Network calculating Q values of each state.
    '''
    def __init__(self,  num_actions, hidden_layer_size=600):
        super(Net, self).__init__()
        self.input_state = 48  # the dimension of state space
        self.num_actions = num_actions  # the dimension of action space
        self.fc1 = nn.Linear(self.input_state, 32*12)  # input layer
        self.fc2 = nn.Linear(32*12, hidden_layer_size)  # hidden layer
        self.fc3 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc4 = nn.Linear(hidden_layer_size, num_actions)  # output layer

    def forward(self, states):
        '''
        Forward the state to the neural network.        
        Parameter:
            states: a batch size of states
        Return:
            q_values: a batch size of q_values
        '''
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_values = self.fc4(x)
        return q_values


class Agent():
    def __init__(self, env, epsilon=0, learning_rate=0.0002, GAMMA=0.97, batch_size=32, capacity=10000):
        """
        The agent learning how to control the action of the cart pole.
        Hyperparameters:
            epsilon: Determines the explore/expliot rate of the agent
            learning_rate: Determines the step size while moving toward a minimum of a loss function
            GAMMA: the discount factor (tradeoff between immediate rewards and future rewards)
            batch_size: the number of samples which will be propagated through the neural network
            capacity: the size of the replay buffer/memory
        """
        self.env = env
        self.n_actions = 2  # the number of actions
        self.count = 0

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.batch_size = batch_size
        self.capacity = capacity

        self.buffer = replay_buffer(self.capacity)
        self.evaluate_net = Net(self.n_actions)  # the evaluate network
        self.evaluate_net = self.evaluate_net.to('cuda')
        self.target_net = Net(self.n_actions)  # the target network
        self.target_net = self.target_net.to('cuda')
        self.optimizer = torch.optim.Adam(
            self.evaluate_net.parameters(), lr=self.learning_rate)  # Adam is a method using to optimize the neural network

    def learn(self):
        '''
        Steps:
        1. Update target net by current net every 100 times. (we have done this for you)
        2. Sample trajectories of batch size from the replay buffer.
        3. Forward the data to the evaluate net and the target net.
        4. Compute the loss with MSE.
        5. Zero-out the gradients.
        6. Backpropagation.
        7. Optimize the loss function.
        '''
        if self.count % 10 == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.tensor(np.array(states), dtype=torch.float).to('cuda')
        actions = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(-1).to('cuda')
        rewards = torch.tensor(np.array(rewards), dtype=torch.float).to('cuda')
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to('cuda')
        dones = torch.tensor(np.array(dones), dtype=torch.float).to('cuda')
        
        q_values = torch.gather(self.evaluate_net(states), 1, actions)
        
        next_actions = self.evaluate_net(next_states).argmax(dim=1, keepdim=True)
        next_q_values = self.target_net(next_states).gather(1, next_actions).reshape(32)
        target_q_values = (rewards + self.gamma * (1 - dones) * next_q_values).unsqueeze(1)

        loss = F.mse_loss(q_values, target_q_values)
        
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()
        
        


    def choose_action(self, state):
        with torch.no_grad():
            temp = np.random.random()
            if temp < math.exp(-1*self.epsilon) or temp<0.005:
                return np.random.randint(self.n_actions)
            action = torch.argmax(self.evaluate_net(Tensor(state).reshape(48).to('cuda'))).item()
        return action


def train(env):
    agent = Agent(env)
    episode = 1000
    rewards = []
    cnt = 0
    for e in range(episode):
        cnt += 1
        state = env.reset()
        #print(state)
        count0 = 0
        count1 = 0
        while True:
            agent.count += 1
            #env.render()
            tempstate1 = state.reshape(48)
            state = state.reshape(48)
            for i in range(12):
                for j in range(4):
                    tempstate1[i*4+j] = (state[44+j] - state[4*i+j])/state[44+j]
            action = agent.choose_action(tempstate1)
            next_state, reward, done, _ = env.step(action)
            tempstate2 = next_state.reshape(48)
            next_state = next_state.reshape(48)
            for i in range(12):
                for j in range(4):
                    tempstate2[i*4+j] = (next_state[44+j] - next_state[4*i+j])/next_state[44+j]
            agent.buffer.insert(tempstate1, int(action), reward, tempstate2, int(done))
            if(action==1):
                count1 += 1
            else:
                count0 += 1
            if len(agent.buffer) >= 100:
                agent.learn()
            if done:
                rewards.append(env._total_reward)
                #print("!")
                #print(count0)
                #print(count1)
                #print(agent.env._total_reward)
                #print(agent.env._total_profit)
                break
            state = next_state
        print(e, env._total_reward, env._total_profit)
        agent.epsilon += 0.1
        
        if(cnt % 999 ==0):
            url = "Tables/DDQN_oo.pt"
            url2 = "Rewards/DDQN_rewards_oor.npy"
            try:
                np.save(url2, np.array(rewards))
            except RuntimeError:
                print("!!")  
            try:
                torch.save(agent.target_net.state_dict(), url)
            except RuntimeError:
                print("!!!")

def test(env):
    """
    Test the agent on the given environment.
    Paramenters:
        env: the given environment.
    Returns:
        None (Don't need to return anything)
    """
    w = SummaryWriter('tb_record_1/comp_profit_train/DDQN')
    testing_agent = Agent(env)
    testing_agent.target_net.load_state_dict(torch.load("Tables/DDQN.pt"))
    for _ in range(1):
        state = env.reset().reshape(48)
        t = 0
        while True:
            tempstate = state
            for i in range(12):
                for j in range(4):
                    tempstate[i*4+j] = (state[44+j] - state[i*4+j])/state[44+j]
            Q = testing_agent.target_net(
                torch.FloatTensor(tempstate.reshape(48)).to('cuda')).squeeze(0).cpu().detach()
            action = int(torch.argmax(Q).numpy())
            next_state, _, done, _ = env.step(action)
            w.add_scalar('Profit', env._total_profit, t)
            t+=1
            if done:
                break
            state = next_state.reshape(48)
    print(env._total_profit)
    print(env._total_reward)
    #env.render()
    #env.save_rendering('Images/DDQN.png')

def state_preprocess(state):
    tempstate = state
    for i in range(12):
        for j in range(4):
            tempstate[i*4+j] = (state[44+j] - state[i*4+j])/state[44+j]
    return tempstate

def gen_offline_data(episodes, env):
    agent = Agent(env)
    agent.target_net.load_state_dict(torch.load("./Tables/DDQN.pt"))
    episode_data = {}
    for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
        episode_data[k] = []
    for e in range(episodes):
        observation = env.reset()
        env.seed(e)
        observation = state_preprocess(observation.reshape(-1))
        while True:
            Q = agent.target_net.forward(torch.FloatTensor(observation).to('cuda')).squeeze(0).cpu().detach()
            action = int(torch.argmax(Q))
            next_observation, reward, done, info = env.step(action)
            episode_data['observations'].append(np.array(observation))
            next_observation = state_preprocess(next_observation.reshape(-1))
            episode_data['next_observations'].append(np.array(next_observation))
            episode_data['actions'].append(np.array(Q))
            episode_data['rewards'].append(np.array([reward]).astype('float32'))
            episode_data['terminals'].append(np.array([done]))
            observation = next_observation
            if done:
                print(info)
                break
    return episode_data

if __name__ == "__main__":
    env = buildEnv.createEnv(2330, frame_bounds=(12,1000))        
    os.makedirs("./Tables", exist_ok=True)
    os.makedirs("./Rewards", exist_ok=True)
    # training section:
    
    #train(env)
        
    # testing section:
    test(env)
    #env.close()
    #np.save("./Rewards/DDQN_rewards.npy", np.array(total_rewards))