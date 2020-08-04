import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import torch.optim as optim


import numpy as np

class Policy():
    def __init__(self, *, env):

        self.net = MlpDiscrete(env.observation_space.shape[0], env.action_space.n)

    def act(self, obs):
        return self.net.act(obs)

    def evaluate(self, obs, actions):
        return self.net.evaluate(obs, actions)

    def parameters(self):
        return self.net.parameters()

class MlpDiscrete(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 128):
        super(MlpDiscrete, self).__init__()

        self.actor = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size), nn.ReLU()
                                    )

        
        self.critic = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size), nn.ReLU()
                                    )

        self.a1 = nn.Linear(hidden_size, hidden_size)
        self.a2 = nn.Linear(hidden_size, output_size)

        self.c1 = nn.Linear(hidden_size, hidden_size)
        self.c2 = nn.Linear(hidden_size, 1)


    def forward(self, x):
        actor = self.actor(x)
        actor = F.relu(self.a1(actor))
        actor = F.tanh(self.a2(actor))

        critic = self.critic(x)
        critic = F.relu(self.c1(critic))
        critic = self.c2(critic)

        return actor, critic

    def __call__(self, x):
        return self.forward(x)

    def act(self, obs):
        obs = torch.FloatTensor(obs)
        logits, state_values = self(obs)
        dist = distributions.Categorical(F.softmax(logits, dim = -1))

        state_values = state_values.squeeze()
        actions = dist.sample()
        action_log_probs = dist.log_prob(actions)

        return actions, state_values, action_log_probs

    def evaluate(self, obs, actions):
        obs = torch.FloatTensor(obs)
        logits, state_values = self(obs)
        dist = distributions.Categorical(F.softmax(logits, dim = -1))

        state_values = state_values.squeeze()

        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        return state_values, action_log_probs, dist_entropy


class Actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 128):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size), nn.ReLU()
                                    )


        self.a1 = nn.Linear(hidden_size, hidden_size)
        self.a2 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        actor = self.actor(x)
        actor = F.relu(self.a1(actor))
        actor = F.tanh(self.a2(actor))

        return actor

    def act(self, obs):
        obs = torch.FloatTensor(obs)
    




                