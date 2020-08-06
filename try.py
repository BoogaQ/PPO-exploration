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
    env = make_vec_env("Pendulum-v0", 4, vec_env_cls = SubprocVecEnv)
    env = VecNormalize(env)

    model = PPO(env = env, lr = 0.0003, nstep = 128, batch_size = 128, max_grad_norm = 1, 
                    hidden_size = 32)
    model.learn(total_timesteps = 1e+7, log_interval = 10)


