import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import torch.optim as optim


import numpy as np

class Policy():
    def __init__(self, state_dim, action_dim, hidden_size):

        self.net = MlpIntDiscrete(state_dim, action_dim, hidden_size = hidden_size)

    def act(self, obs):
        return self.net.act(obs)

    def evaluate(self, obs, actions):
        return self.net.evaluate(obs, actions)

    def parameters(self):
        return self.net.parameters()


class Swish(nn.Module):
    def forward(self, x):
        return x * nn.Sigmoid(x)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class MlpDiscrete(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 128):
        super(MlpDiscrete, self).__init__()

        self.actor = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size), nn.ReLU()
                                    )

        
        self.critic = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size), nn.ReLU()
                                    )

        self.a1 = nn.Linear(hidden_size, output_size)
        self.c1 = nn.Linear(hidden_size, 1)


    def forward(self, x):
        actor = self.actor(x)
        actor = self.a1(actor)

        critic = self.critic(x)
        critic = self.c1(critic)

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


class MlpContinuous(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 128):
        super(MlpContinuous, self).__init__()

        self.actor = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size), nn.ReLU()
                                    )

        
        self.critic = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size), nn.ReLU()
                                    )

        self.action_log_std = nn.Parameter(torch.zeros(1, output_size))

        self.a1 = nn.Linear(hidden_size, output_size)
        self.c1 = nn.Linear(hidden_size, 1)


    def forward(self, x):
        actor = self.actor(x)
        mean = self.a1(actor)

        action_log_std = self.action_log_std

        critic = self.critic(x)
        critic = self.c1(critic)

        return mean, action_log_std, critic

    def __call__(self, x):
        return self.forward(x)

    def act(self, obs):
        obs = torch.FloatTensor(obs)
        mean, action_log_std, state_values = self(obs)
        dist = distributions.Normal(mean, torch.exp(action_log_std))

        state_values = state_values.squeeze()

        actions = dist.sample()

        action_log_probs = dist.log_prob(actions)

        return actions, state_values, action_log_probs

    def evaluate(self, obs, actions):
        obs = torch.FloatTensor(obs)

        mean, action_log_std, state_values = self(obs)
        dist = distributions.Normal(mean, torch.exp(action_log_std))

        state_values = state_values.squeeze()

        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        return state_values, action_log_probs, dist_entropy


class MlpIntDiscrete(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 32):
        super(MlpIntDiscrete, self).__init__()

        self.actor = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size), nn.ReLU()
                                    )

        
        self.critic = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size), nn.ReLU()
                                    )

        self.int_critic = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                                        nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                        nn.Linear(hidden_size, hidden_size), nn.ReLU()
                                        )

        self.a1 = nn.Linear(hidden_size, output_size)

        self.c1 = nn.Linear(hidden_size, 1)
        self.int_c1 = nn.Linear(hidden_size, 1)


    def forward(self, x):
        actor = self.actor(x)
        actor = self.a1(actor)

        critic = self.critic(x)
        critic = self.c1(critic)

        int_critic = self.critic(x)
        int_critic = self.c1(int_critic)


        return actor, critic, int_critic

    def __call__(self, x):
        return self.forward(x)

    def act(self, obs):
        obs = torch.FloatTensor(obs)
        logits, state_values, int_state_values = self(obs)
        dist = distributions.Categorical(F.softmax(logits, dim = -1))

        state_values = state_values.squeeze()
        int_state_values = int_state_values.squeeze()
        actions = dist.sample()
        action_log_probs = dist.log_prob(actions)

        return actions, state_values, int_state_values, action_log_probs

    def evaluate(self, obs, actions):
        obs = torch.FloatTensor(obs)
        logits, state_values, int_state_values = self(obs)
        dist = distributions.Categorical(F.softmax(logits, dim = -1))

        state_values = state_values.squeeze()
        int_state_values = int_state_values.squeeze()

        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        return state_values, int_state_values, action_log_probs, dist_entropy


class RndNetwork(nn.Module):
    def __init__(self, input_size, hidden_size = 32):
        super(RndNetwork, self).__init__()
        
        self.predictor = nn.Sequential(nn.Linear(input_size, hidden_size),
                                                 nn.Tanh(),
                                                 nn.Linear(hidden_size, hidden_size),
                                                 nn.Tanh(),
                                                 nn.Linear(hidden_size, 1)
                                                 )
        
        self.target = nn.Sequential(nn.Linear(input_size, hidden_size),
                                    nn.ELU(),
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ELU(),
                                    nn.Linear(hidden_size, 1)
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
        return predict, target

    def __call__(self, x):
        return self.forward(x)

    def int_reward(self, obs):
        obs = torch.FloatTensor(obs)
        
        pred, target = self(obs)
        int_rew = torch.mean(F.mse_loss(pred, target, reduction='none'))
        
        return int_rew.data.to('cpu').numpy()


class Actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 16):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size), nn.ReLU()
                                    )


        self.a1 = nn.Linear(hidden_size, hidden_size)
        self.a2 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        actor = self.actor(x)
        actor = F.relu(self.a1(actor))
        actor = self.a2(actor)
        return actor

    def act(self, obs):
        obs = torch.FloatTensor(obs)
        logits = self.forward(obs)
        dist = distributions.Categorical(F.softmax(logits, dim = -1))
        return dist.sample()


    
