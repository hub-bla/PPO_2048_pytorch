import torch
from torch import nn
import numpy as np
from utils import init_layer
from torch.distributions.categorical import Categorical

class Critic(nn.Module):
    def __init__(self, observation_shape):
        super().__init__()
        self.linear1 = nn.Linear(16, 64)
        self.batch_norm = nn.BatchNorm1d(64)
        self.linear2 = nn.Linear(64,64)
        self.linear5 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        # x =self.batch_norm(x)
        # x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        x = torch.nn.functional.tanh(x)
        x = self.linear5(x)
        x = torch.nn.functional.tanh(x)
        return x






class Actor(nn.Module):
    def __init__(self, observation_shape, length_action_space):
        super().__init__()
        self.linear1 = nn.Linear(16, 64)
        self.batch_norm = nn.BatchNorm1d(64)
        self.linear2 = nn.Linear(64,64)
        self.linear5 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.linear1(x)
        # x =self.batch_norm(x)
        x = self.linear2(x)
        x = torch.nn.functional.relu(x)
        x = self.linear5(x)
 
        x = nn.functional.log_softmax(x, dim=-1)
        return x
    


class PpoAgent(nn.Module):
    def __init__(self, observation_shape, length_action_space):
        super().__init__()
        self.critic = Critic(observation_shape)
        self.actor = Actor(observation_shape, length_action_space)

    
    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

