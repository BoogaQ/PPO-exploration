import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions


from gym.spaces import Box, Discrete


class Agent():
    def __init__(self, *, env, device = 'cpu'):
        self.action_type = env.action_space.__class__.__name__
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n if self.action_type == "Discrete" else env.action_space.shape[0]

        if self.action_type == "Discrete":
            self.dist = distributions.Categorical
        elif self.action_type == "Box":
            self.dist = distributions.Normal
        else:
            raise NotImplementedError()

        if len(env.observation_space.shape) == 1:
            self.net = MlpNetwork(self.state_size, self.action_size).to(device)
        elif len(env.observation_space.shape) == 3:
            self.net = CnnNetwork(4, self.action_size).to(device)  
        else:
            raise NotImplementedError() 

    def act(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)

        if self.action_type == "Discrete":
            logits, _, state_values = self.net(obs)
            dist = self.dist(F.softmax(logits, dim = -1))
        elif self.action_type == "Box":
            mean, std_sq, state_values = self.net(obs)
            dist = self.dist(mean, std_sq.sqrt())

        state_values = state_values.squeeze()
        actions = dist.sample()

        action_log_probs = dist.log_prob(actions)

        return actions, state_values, action_log_probs

    def evaluate(self, obs, actions):
        obs = torch.FloatTensor(obs).to(self.device)

        if self.action_type == "Discrete":
            logits, _, state_values = self.net(obs)
            dist = self.dist(F.softmax(logits, dim = -1))
        elif self.action_type == "Box":
            mean, std_sq, state_values = self.net(obs)
            dist = self.dist(mean, std_sq.sqrt())

        action_log_probs = dist.log_prob(actions.to(self.device))
        dist_entropy = dist.entropy()

        state_values = state_values.squeeze()
        action_log_probs = action_log_probs.squeeze() 

        return state_values, action_log_probs, dist_entropy

    def parameters(self):
        return self.net.parameters()
