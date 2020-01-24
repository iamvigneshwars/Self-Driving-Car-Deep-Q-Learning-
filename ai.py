# Importing the libraries

import random 
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from  torch.autograd import Variable

# Architecture of Neural Net

class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_actions = nb_actions
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# Memory replay

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []


    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size)
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
