import argparse
from collections import deque
import itertools
import random
import time
import cv2

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class CarRacingEnvironment:
	def __init__(self, N_frame=4, test=False):
		self.test = test
		if self.test:
			self.env = gym.make('CarRacing-v2', render_mode="human")
		else:
			self.env = gym.make('CarRacing-v2')
		self.action_space = self.env.action_space
		self.observation_space = self.env.observation_space
		self.ep_len = 0
		self.frames = deque(maxlen=N_frame)
	
	def check_car_position(self, obs):
		# cut the image to get the part where the car is
		part_image = obs[60:84, 40:60, :]

		road_color_lower = np.array([90, 90, 90], dtype=np.uint8)
		road_color_upper = np.array([120, 120, 120], dtype=np.uint8)
		grass_color_lower = np.array([90, 180, 90], dtype=np.uint8)
		grass_color_upper = np.array([120, 255, 120], dtype=np.uint8)
		road_mask = cv2.inRange(part_image, road_color_lower, road_color_upper)
		grass_mask = cv2.inRange(part_image, grass_color_lower, grass_color_upper)
		# count the number of pixels in the road and grass
		road_pixel_count = cv2.countNonZero(road_mask)
		grass_pixel_count = cv2.countNonZero(grass_mask)

		# save image for debugging
		# filename = "images/image" + str(self.ep_len) + ".jpg"
		# cv2.imwrite(filename, part_image)

		return road_pixel_count, grass_pixel_count

	def step(self, action):
		obs, reward, terminates, truncates, info = self.env.step(action)
		original_reward = reward
		original_terminates = terminates
		self.ep_len += 1
		road_pixel_count, grass_pixel_count = self.check_car_position(obs)
		info["road_pixel_count"] = road_pixel_count
		info["grass_pixel_count"] = grass_pixel_count

		# my reward shaping strategy, you can try your own
		if road_pixel_count < 10:
			terminates = True
			reward = -100

		# convert to grayscale
		obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY) # 96x96

		# save image for debugging
		# filename = "images/image" + str(self.ep_len) + ".jpg"
		# cv2.imwrite(filename, obs)

		# frame stacking
		self.frames.append(obs)
		obs = np.stack(self.frames, axis=0)

		if self.test:
			# enable this line to recover the original reward
			reward = original_reward
			# enable this line to recover the original terminates signal, disable this to accerlate evaluation
			# terminates = original_terminates

		return obs, reward, terminates, truncates, info
	
	def reset(self):
		obs, info = self.env.reset(seed=0)	# 1: 518, 2: 244, 3:877, 4:867
		self.ep_len = 0
		obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY) # 96x96

		# frame stacking
		for _ in range(self.frames.maxlen):
			self.frames.append(obs)
		obs = np.stack(self.frames, axis=0)

		return obs, info
	
	def render(self):
		self.env.render()
	
	def close(self):
		self.env.close()

if __name__ == '__main__':
	env = CarRacingEnvironment(test=True)
	obs, info = env.reset()
	done = False
	total_reward = 0
	total_length = 0
	t = 0
	while not done:
		t += 1
		action = env.action_space.sample()
		action[2] = 0.0
		obs, reward, terminates, truncates, info = env.step(action)
		print(f'{t}: road_pixel_count: {info["road_pixel_count"]}, grass_pixel_count: {info["grass_pixel_count"]}, reward: {reward}')
		total_reward += reward
		total_length += 1
		env.render()
		if terminates or truncates:
			done = True

	print("Total reward: ", total_reward)
	print("Total length: ", total_length)
	env.close()
