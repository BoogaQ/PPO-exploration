import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from models import *
from ppo import PPO, PPO_RND, PPO_ICM
from buffer import RolloutStorage
from env import *

import torch
import torch.distributions as distributions
import torch.nn.functional as F
import torch.optim
import numpy as np

import gym
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import vec_normalize
import pybulletgym

if __name__ == "__main__":
    model = PPO_RND(env_id = "BipedalWalker-v3", lr = 0.0003, vf_coef = 1, nstep = 128, int_vf_coef = 2, 
                    batch_size = 128, hidden_size = 64, n_epochs = 4, int_hidden_size = 256, rnd_start = 1e+4)

    model.learn(total_timesteps = 1e+7, log_interval = 1, reward_target = 200)


