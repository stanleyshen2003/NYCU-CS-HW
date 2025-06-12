import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from gae_replay_buffer import GaeSampleMemory
from replay_buffer import ReplayMemory
from abc import ABC, abstractmethod
import cv2

class PPOBaseAgent(ABC):
	def __init__(self, config):
		self.gpu = config["gpu"]
		self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
		self.total_time_step = 0
		self.training_steps = int(config["training_steps"])
		self.update_sample_count = int(config["update_sample_count"])
		self.discount_factor_gamma = config["discount_factor_gamma"]
		self.discount_factor_lambda = config["discount_factor_lambda"]
		self.clip_epsilon = config["clip_epsilon"]
		self.max_gradient_norm = config["max_gradient_norm"]
		self.batch_size = int(config["batch_size"])
		self.value_coefficient = config["value_coefficient"]
		self.entropy_coefficient = config["entropy_coefficient"]
		self.eval_interval = config["eval_interval"]
		self.eval_episode = config["eval_episode"]
		self.eval_frame = deque(maxlen=5)
		self.eval_reset = False
		self.count = 0
		self.gae_replay_buffer = GaeSampleMemory({
			"horizon" : config["horizon"],
			"use_return_as_advantage": False,
			"agent_count": 1,
			})

		self.writer = SummaryWriter(config["logdir"])

	@abstractmethod
	def decide_agent_actions(self, observation):
		# add batch dimension in observation
		# get action, value, logp from net

		return NotImplementedError

	@abstractmethod
	def update(self):
		# sample a minibatch of transitions
		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
		# calculate the loss and update the behavior network

		return NotImplementedError


	def train(self):
		episode_idx = 0
		while self.total_time_step <= self.training_steps:
			observation, info = self.env.reset()
			episode_reward = 0
			episode_len = 0
			episode_idx += 1
			while True:
				action, value, logp_pi = self.decide_agent_actions(observation)
				
				next_observation, reward, terminate, truncate, info = self.env.step(self.actions[action])
				#print(observation.shape)
				#print(action.detach().cpu().numpy())
				#print(reward)
				#print(logp_pi.squeeze().detach().cpu().numpy())
				
				#print(value.squeeze().detach().cpu().numpy())
				# print(logp_pi.item())
				#print(reward)
				#print(terminate)
				# observation must be dict before storing into gae_replay_buffer
				# dimension of reward, value, logp_pi, done must be the same
				_, reward2, terminate2, _, _ = self.env.step(self.actions[action])
				terminate = terminate or terminate2
				obs = {}
				obs["observation_2d"] = np.asarray(observation, dtype=np.float32)
				self.gae_replay_buffer.append(0, {
						"observation": obs,    # shape = (4,84,84)
						"action": action.detach().cpu().numpy(),      # shape = (1,)
						"reward": (reward+reward2)*500,      # shape = ()
						"value": value.squeeze().detach().cpu().numpy(),        # shape = ()
						"logp_pi": logp_pi.squeeze().detach().cpu().numpy(),    # shape = ()
						"done": terminate,     # shape = ()
					})

				if len(self.gae_replay_buffer) >= self.update_sample_count:
					self.update()
					self.gae_replay_buffer.clear_buffer()

				episode_reward += reward + reward2
				episode_len += 1
				
				if terminate or truncate:
					self.writer.add_scalar('Train/Episode Reward', episode_reward, self.total_time_step)
					self.writer.add_scalar('Train/Episode Len', episode_len, self.total_time_step)
					print(f"[{len(self.gae_replay_buffer)}/{self.update_sample_count}][{self.total_time_step}/{self.training_steps}]  episode: {episode_idx}  episode reward: {episode_reward}  episode len: {episode_len}")
					break
					
				observation = next_observation
				self.total_time_step += 1
				
			if episode_idx % self.eval_interval == 0:
				# save model checkpoint
				avg_score = self.evaluate()
				self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score*100)}.pth"))
				self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)

	def evaluate(self):
		print("==============================================")
		print("Evaluating...")
		all_rewards = []
		for i in range(self.eval_episode):
			observation, info = self.test_env.reset()
			
			#self.test_env.seed(0)
			total_reward = 0
			while True:
				#self.test_env.render()
				action, _, _ = self.decide_agent_actions(observation, eval=True)
				next_observation, reward, terminate, truncate, info = self.test_env.step(self.actions[action])
				total_reward += reward
				if terminate or truncate:
					print(f"episode {i+1} reward: {total_reward}")
					all_rewards.append(total_reward)
					break

				observation = next_observation
			

		avg = sum(all_rewards) / self.eval_episode
		print(f"average score: {avg}")
		print("==============================================")
		return avg
	
	# save model
	def save(self, save_path):
		torch.save(self.net.state_dict(), save_path)

	# load model
	def load(self, load_path):
		self.net.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))

	# load model weights and evaluate
	def load_and_evaluate(self, load_path):
		self.load(load_path)
		self.evaluate()

	def act(self, obs):
		obs = np.transpose(obs, (1, 2, 0))
		obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
		obs = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_AREA)
		#print(obs.shape)
		if not self.eval_reset:
			for i in range(5):
				temp = deque(maxlen=4)
				for _ in range(4):
					temp.append(obs)
				self.eval_frame.append(temp)
		else:
			temp = self.eval_frames[0]
			temp.append(obs)

		obs = np.stack(temp, axis=0)
		#print(obs.shape)
		action,_,_ = self.decide_agent_actions(obs, eval=True)
		#np.array([action[0] - action[2], action[1]])
		return np.array(self.actions[action])


	

