import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import torch.optim as optim

from buffer import RolloutStorage

import numpy as np

class Policy():
    def __init__(self, *, env, device = 'cpu', base = 'Cnn'):
        self.device = device
        if base == 'Cnn':
            self.base = CnnActorCritic(4, env.action_space.n).to(device)  

    def act(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        logits, state_values = self.base(obs)
        dist = distributions.Categorical(F.softmax(logits, dim = 1))

        state_values = state_values.squeeze(1)
        actions = dist.sample()
        action_log_probs = dist.log_prob(actions)

        return actions, state_values, action_log_probs

    def evaluate(self, obs, actions):
        obs = torch.FloatTensor(obs).to(self.device)
        logits, state_values = self.base(obs)
        dist = distributions.Categorical(F.softmax(logits, dim = 1))

        state_values = state_values.squeeze(1)

        action_log_probs = dist.log_prob(actions.to(self.device))
        dist_entropy = dist.entropy()

        return state_values, action_log_probs, dist_entropy

    def parameters(self):
        return self.base.parameters()


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(1), -1)
    
class CnnActorCritic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 512):
        super(CnnActorCritic, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_size, 32, kernel_size = 8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, hidden_size),
            nn.ReLU()
            )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, output_size)
        )

        self.extra_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.critic_ext = nn.Linear(hidden_size, 1)
        
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, np.sqrt(2))
                module.bias.data.fill_(0.0)     
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)

        x = self.feature_extractor(x)
        policy = self.actor(x)
        value_ext = self.critic_ext(self.extra_layer(x))
        
        return policy, value_ext
    
    def __call__(self, x):
        return self.forward(x)
        
    
class RndNetwork(nn.Module):
    def __init__(self, input_size):
        super(RndNetwork, self).__init__()
        
        self.predictor = nn.Sequential(
            nn.Conv2d(input_size, 32, kernel_size = 8, stride = 4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size = 2, stride = 1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
            )
        
        self.target = nn.Sequential(
            nn.Conv2d(input_size, 32, kernel_size = 8, stride = 4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size = 2, stride = 1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512)
            )
        
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, np.sqrt(2))
                module.bias.data.zero_()
                
        for param in self.target.parameters():
            param.requires_grad = False
            
    def forward(self, x):      
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)

        predict = self.predictor(x)
        target = self.target(x)
        return predictor, target

    def __call__(self, x):
        print(x)
        return self.forward(x)
        
        