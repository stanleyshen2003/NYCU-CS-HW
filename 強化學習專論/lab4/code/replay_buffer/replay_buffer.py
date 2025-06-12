import numpy as np
import torch
from collections import deque
import random

class ReplayMemory(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        """Saves a transition"""
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        """Sample a batch of transitions"""
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(np.asarray(x), dtype=torch.float, device=device) for x in zip(*transitions))
    