import argparse
from collections import deque
import itertools
import random
import time
import cv2

from racecar_gym.env import RaceEnv
import numpy as np
import torch
import torch.nn as nn
#from torch.utils.tensorboard import SummaryWriter


class CarRacingEnvironment:
	def __init__(self, N_frame=4, test=False):
		self.test = test # set to false
		self.env = RaceEnv(
			scenario='austria_competition_collisionStop',
			render_mode='rgb_array_birds_eye',
			reset_when_collision=True,
		)
		self.action_space = self.env.action_space
		self.observation_space = self.env.observation_space
		self.ep_len = 0
		self.max_batch = 5
		self.N_frame = N_frame
		self.frames = deque(maxlen=self.max_batch)
		

	def step(self, action):
		#print(action)
		obs, reward, terminates, truncates, info = self.env.step(action)
		#print(info)
		original_reward = reward
		#print(info["reward"])
		self.ep_len += 1
		if info["wrong_way"]:
			reward = -0.005
		
		velocity = info['velocity'][0] **2 + info['velocity'][1]**2
		if info["wall_collision"] and ((info["progress"] > 0.36 and info["progress"] < 0.37)):
			reward = -0.03
			print("collide")
			print(info["progress"])
		elif info["wall_collision"]:
			reward = -0.01
			print("collide")
			print(info["progress"])
		#elif len(info["opponent_collisions"]) != 0:
		#	reward = -0.01
		#elif (velocity > 5 or velocity < 0.5):
		#	reward = - 0.0006
		#print(velocity)
		if self.test:
			reward = original_reward

		obs = np.transpose(obs, (1, 2, 0))
		obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
		obs = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_AREA)
		
		#frame stacking
		temp = self.frames[0]
		temp.append(obs)
		self.frames.append(temp)
		obs = np.stack(temp, axis=0)

		return obs, reward, terminates, truncates, info
	
	def reset(self):
		kwargs = {}
		if kwargs.get('options'):
			kwargs['options']['mode'] = 'random'
		else:
			kwargs['options'] = {'mode': 'random'}
		obs, info = self.env.reset()	# 3219, 6728, 8844, 7022, 2713
		self.ep_len = 0
		# now the obs is 3,128,128, i want to convert it to 128,128,3
		obs = np.transpose(obs, (1, 2, 0))
		obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY) # 96x96
		obs = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_AREA)
		#filename = "images/image" + str(self.ep_len) + ".jpg"
		#cv2.imwrite(filename, obs)
		#print("saved to", filename)
		# frame stacking
		for i in range(5):
			temp = deque(maxlen=self.N_frame)
			for _ in range(self.N_frame):
				temp.append(obs)
			self.frames.append(temp)
		#return first element in the deque
		obs = np.stack(self.frames[0], axis=0)
		
		return obs, info
	
	def render(self):
		self.env.render()
	
	def close(self):
		self.env.close()

if __name__ == '__main__':
	env = CarRacingEnvironment(test=True)
	obs, info = env.reset()
	print(info['pose'])
	done = False
	total_reward = 0
	total_length = 0
	t = 0
	while not done:
		t += 1
		action = env.action_space.sample()
		print(t)
		action[0] = 1
		obs, reward, terminates, truncates, info = env.step(action)
		total_reward += reward
		total_length += 1
		env.render()
		if terminates or truncates:
			done = True

	print("Total reward: ", total_reward)
	print("Total length: ", total_length)
	env.close()
