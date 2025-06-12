import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent
from atari_model import AtariNetDQN
import gym
import random

from gym.wrappers import GrayScaleObservation, FrameStack, ResizeObservation

class AtariDDQNAgent(DQNBaseAgent):
	def __init__(self, config):
		super(AtariDDQNAgent, self).__init__(config)
		### TODO ###
		# initialize env
		self.env = gym.make(config["env_id"])
		self.env.seed(0)
		self.env = GrayScaleObservation(self.env)
		self.env = ResizeObservation(self.env,84)
		self.env = FrameStack(self.env,4)
		
		### TODO ###
		# initialize test_env
		self.test_env = gym.make(config["env_id"])
		self.test_env.seed(80000)
		self.test_env = GrayScaleObservation(self.test_env)
		self.test_env = ResizeObservation(self.test_env,84)
		self.test_env = FrameStack(self.test_env,4)

		# initialize behavior network and target network
		self.behavior_net = AtariNetDQN(self.env.action_space.n)
		self.behavior_net.to(self.device)
		self.target_net = AtariNetDQN(self.env.action_space.n)
		self.target_net.to(self.device)
		self.target_net.load_state_dict(self.behavior_net.state_dict())
		# initialize optimizer
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)
		
	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		### TODO ###
		# get action from behavior net, with epsilon-greedy selection
		#print(observation.shape)
		# Convert the frame to a PyTorch tensor and preprocess it as needed
		observation_array = np.array(observation, dtype=np.float32)
		frame = torch.tensor(observation_array, dtype=torch.float32).to(self.device)
		#print(frame.squeeze(-1).shape)
		if random.random() < epsilon:
			action = random.randint(0, action_space.n - 1)
		else:
			with torch.no_grad():
				action = np.argmax(self.behavior_net(frame.squeeze(-1).unsqueeze(0))).item()
		return action
	
	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		### TODO ###
		# calculate the loss and update the behavior network
		# 1. get Q(s,a) from behavior net
		# 2. get max_a Q(s',a) from target net
		# 3. calculate Q_target = r + gamma * max_a Q(s',a)
		# 4. calculate loss between Q(s,a) and Q_target
		# 5. update behavior net
		#print(self.behavior_net(state.squeeze(-1)).shape)
		#print(state.squeeze(-1).shape)
		#print(action.shape)
		q_value = self.behavior_net(state.squeeze(-1)).gather(1, action.long())
		with torch.no_grad():
			next_action = torch.argmax(self.behavior_net(next_state.squeeze(-1)),dim=1).unsqueeze(1)
			q_next = self.target_net(next_state.squeeze(-1)).gather(1,next_action)

			# if episode terminates at next_state, then q_target = reward
			q_target = reward + (1 - done) * self.gamma * q_next
		
		
		criterion = torch.nn.MSELoss()
		loss = criterion(q_value, q_target)

		self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)

		self.optim.zero_grad()
		loss.backward()
		self.optim.step()

	