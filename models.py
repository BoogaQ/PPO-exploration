import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import torch.optim as optim

import numpy as np

# https://github.com/uvipen/Street-fighter-A3C-ICM-pytorch
# https://github.com/adik993/ppo-pytorch
# https://github.com/DLR-RM/stable-baselines3
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

class Policy(nn.Module):
    def __init__(self, env, hidden_size, intrinsic_model = False):
        super(Policy, self).__init__()
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_type = self.env.action_space.__class__.__name__
        self.action_dim = self.env.action_space.n if self.action_type == "Discrete" else self.env.action_space.shape[0]

        self.intrinsic = intrinsic_model

        if intrinsic_model:
            self.net = MlpIntrinsic(self.state_dim, self.action_dim, hidden_size = hidden_size)
        else:
            self.net = MlpNetwork(self.state_dim, self.action_dim, hidden_size = hidden_size)

    def act(self, obs):
        if self.intrinsic:
            return self.act_intrinsic(obs)
        obs = torch.FloatTensor(obs)       

        if self.action_type == "Discrete":
            logits, state_values = self.net(obs)
            state_values = state_values.squeeze()
            dist = distributions.Categorical(F.softmax(logits, dim = -1))
            actions = dist.sample().squeeze()
            action_log_probs = dist.log_prob(actions).squeeze()

        elif self.action_type == "Box":
            logits, sd, state_values = self.net.forward_continuous(obs)
            state_values = state_values.squeeze()
            dist = distributions.Normal(logits, torch.exp(sd))
            actions = dist.sample()
            action_log_probs = dist.log_prob(actions)
            dist_entropy = dist.entropy()   
        
        return actions, state_values, action_log_probs

    def evaluate(self, obs, actions):
        if self.intrinsic:
            return self.evaluate_intrinsic(obs, actions)

        obs = torch.FloatTensor(obs)     

        if self.action_type == "Discrete":
            logits, state_values = self.net(obs)
            state_values = state_values.squeeze()
            actions = actions.flatten()
            dist = distributions.Categorical(F.softmax(logits, dim = -1))
            action_log_probs = dist.log_prob(actions).unsqueeze(1)
            dist_entropy = dist.entropy()

        elif self.action_type == "Box":
            logits, sd, state_values = self.net.forward_continuous(obs)
            state_values = state_values.squeeze()
            dist = distributions.Normal(logits, torch.exp(sd))
            action_log_probs = dist.log_prob(actions)
            dist_entropy = dist.entropy()    

        return state_values, action_log_probs, dist_entropy

    def act_intrinsic(self, obs):
        assert self.intrinsic # Only usable with random network distillation

        obs = torch.FloatTensor(obs)

        if self.action_type == "Discrete":
            logits, state_values, int_state_values = self.net(obs)
            state_values = state_values.squeeze()
            int_state_values = int_state_values.squeeze()

            dist = distributions.Categorical(F.softmax(logits, dim = -1))
            actions = dist.sample().squeeze()
            action_log_probs = dist.log_prob(actions).squeeze()

        elif self.action_type == "Box":
            logits, sd, state_values, int_state_values = self.net.forward_continuous(obs)
            state_values = state_values.squeeze()
            int_state_values = int_state_values.squeeze()

            dist = distributions.Normal(logits, torch.exp(sd))
            actions = dist.sample()
            action_log_probs = dist.log_prob(actions)
            dist_entropy = dist.entropy()
        
        return actions, state_values, int_state_values, action_log_probs

    def evaluate_intrinsic(self, obs, actions):
        assert self.intrinsic # Only usable with random network distillation

        obs = torch.FloatTensor(obs)      

        if self.action_type == "Discrete":
            logits, state_values, int_state_values = self.net(obs)
            state_values =  state_values.squeeze()
            int_state_values = int_state_values.squeeze()
            actions = actions.flatten()
            dist = distributions.Categorical(F.softmax(logits, dim = -1))
            action_log_probs = dist.log_prob(actions).unsqueeze(1)
            dist_entropy = dist.entropy()

        elif self.action_type == "Box":
            logits, sd, state_values, int_state_values = self.net.forward_continuous(obs)
            state_values =  state_values.squeeze()
            int_state_values = int_state_values.squeeze()

            dist = distributions.Normal(logits, torch.exp(sd))
            action_log_probs = dist.log_prob(actions)
            dist_entropy = dist.entropy()     

        return state_values, int_state_values, action_log_probs, dist_entropy

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, np.sqrt(2))
                nn.init.constant_(module.bias, 0)


class MlpNetwork(BaseNetwork):
    def __init__(self, input_size, output_size, hidden_size = 128):
        super(MlpNetwork, self).__init__()

        self.actor = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh(),
                                   nn.Linear(hidden_size, hidden_size), nn.Tanh(),
                                   nn.Linear(hidden_size, output_size)
                                    )

        
        self.critic = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh(),
                                    nn.Linear(hidden_size, hidden_size), nn.Tanh(),
                                    nn.Linear(hidden_size, 1)
                                    )                       
                                    
        self.action_log_std = nn.Parameter(torch.zeros(1, output_size))

        self.init_weights()

    def forward(self, x):
        actor = self.actor(x)
        critic = self.critic(x)

        return actor, critic

    def forward_continuous(self, x):
        mean = self.actor(x).tanh()
        log_sd = self.action_log_std.expand_as(mean)
        critic = self.critic(x)

        return mean ,log_sd, critic

    def __call__(self, x):
        return self.forward(x)


class MlpIntrinsic(BaseNetwork):
    def __init__(self, input_size, output_size, hidden_size = 128):
        super(MlpIntrinsic, self).__init__()

        self.actor = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh(),
                                   nn.Linear(hidden_size, hidden_size), nn.Tanh(),
                                   nn.Linear(hidden_size, output_size)
                                    )

        
        self.critic = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh(),
                                    nn.Linear(hidden_size, hidden_size), nn.Tanh(),
                                    nn.Linear(hidden_size, 1)
                                    )                       

        self.int_critic = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh(),
                                        nn.Linear(hidden_size, hidden_size), nn.Tanh(),
                                        nn.Linear(hidden_size, 1)
                                        )

        self.action_log_std = nn.Parameter(torch.zeros(1, output_size))

        self.init_weights()

    def forward(self, x):
        actor = self.actor(x)
        critic = self.critic(x)
        int_critic = self.int_critic(x)

        return actor, critic, int_critic
    
    def forward_continuous(self, x):
        mean = self.actor(x).tanh()
        log_sd = self.action_log_std.expand_as(mean)
        critic = self.critic(x)
        int_critic = self.int_critic(x)

        return mean ,log_sd, critic, int_critic

    def __call__(self, x):
        return self.forward(x)


class RndNetwork(BaseNetwork):
    def __init__(self, input_size, hidden_size = 32):
        super(RndNetwork, self).__init__()
        
        self.predictor = nn.Sequential(nn.Linear(input_size, hidden_size),
                                                 nn.ELU(),
                                                 nn.Linear(hidden_size, hidden_size),
                                                 nn.ELU(),
                                                 nn.Linear(hidden_size, 1)
                                                 )
        
        self.target = nn.Sequential(nn.Linear(input_size, hidden_size),
                                    nn.ELU(),
                                    nn.Linear(hidden_size, 1)
                                    )

        for name, param in self.target.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 1)
            elif 'weight' in name:
                nn.init.constant_(param, 0.01)

        for name, param in self.predictor.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.01)
            elif 'weight' in name:
                nn.init.constant_(param, 1)
                
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
        int_rew = (pred - target).pow(2).squeeze()
        
        return int_rew


class IntrinsicCuriosityModule(BaseNetwork):
    def __init__(self, input_size, action_converter, hidden_size):
        super(IntrinsicCuriosityModule, self).__init__()
  
        self.action_converter = action_converter
        self.feature_size = hidden_size
        self.input_size = input_size

        self.output_size = action_converter.action_output
        self.n_actions = action_converter.num_actions

        self.state_encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LeakyReLU(),
                                           nn.Linear(hidden_size, hidden_size), nn.LeakyReLU())

        self.forward_model = nn.Sequential(nn.Linear(self.n_actions + hidden_size, hidden_size),
                                           nn.LeakyReLU(),
                                           nn.Linear(hidden_size, hidden_size))

        self.inverse_model = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size),
                                           nn.LeakyReLU(),
                                           nn.Linear(hidden_size, self.n_actions))
        
        if action_converter.action_type == "Discrete":
            self.action_encoder = nn.Embedding(self.n_actions, self.n_actions)
        else:
            self.action_encoder = nn.Linear(self.n_actions, self.output_size)

        self.init_weights()

    def forward(self, state, next_state, action):
        action = self.action_encoder(action.float() if self.action_converter.action_type == "Box" else action.squeeze().long())

        state_ft = self.state_encoder(state)
        next_state_ft = self.state_encoder(next_state)
        next_state_ft = next_state_ft.view(-1, self.feature_size)

        action_hat = self.inverse_model(torch.cat((state_ft, next_state_ft), 1))
        next_state_hat = self.forward_model(torch.cat((state_ft, action), 1))
        return action_hat, next_state_hat, next_state_ft

    def int_reward(self, state, next_state, action):
        action = self.action_encoder(action.float() if self.action_converter.action_type == "Box" else action.long())

        state_ft = self.state_encoder(state)
        next_state_ft = self.state_encoder(next_state)

        next_state_hat = self.forward_model(torch.cat((state_ft, action), 1))

        int_rew = (next_state_hat - next_state_ft).pow(2).mean(dim = -1)
        return torch.clamp(int_rew, -1, 1)
