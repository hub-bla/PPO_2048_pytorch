import torch
from torch import nn
from torch.distributions.categorical import Categorical


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(16, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 2048, kernel_size=3)
        self.linear1 = nn.Linear(2048*int(2048/512), 64)
        self.linear2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        
        x = x.flatten(start_dim=1)
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        # x = torch.nn.functional.tanh(x)
        return x


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(16, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 2048, kernel_size=3)
        self.linear1 = nn.Linear(2048*int(2048/512), 64)
        self.linear2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        
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
    
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)