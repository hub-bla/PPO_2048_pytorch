import torch
from torch import nn
from torch.distributions.categorical import Categorical
import numpy as np

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(16, 256, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.linear1 = nn.Linear(1024, 64)
        self.linear2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.batch_norm2(x)
        x = x.flatten(start_dim=1)
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        # x = torch.nn.functional.tanh(x)
        return x


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(16, 256, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.linear1 = nn.Linear(1024, 64)
        self.linear2 = nn.Linear(64, 4)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.batch_norm2(x)
        x = x.flatten(start_dim=1)
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        x = nn.functional.log_softmax(x, dim=-1)
        return x

class PpoAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = Critic()
        self.actor = Actor()

    
    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None, legal_actions=None):
        if legal_actions is None:
            legal_actions = torch.ones(x.shape[0], 4)
        logits = torch.where(legal_actions==0.0, -np.infty, self.actor(x))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)