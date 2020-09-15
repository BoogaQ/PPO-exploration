import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from models import *
from algorithms import PPO, PPO_RND, PPO_ICM
from buffer import RolloutStorage
from env import *

import torch
import torch.distributions as distributions
import torch.nn.functional as F
import torch.optim
import numpy as np

from evolution_strategies import EvolutionStrategy

import gym
import pybullet_envs
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import vec_normalize
import pybulletgym
import mujoco_py
from hyperparameters import *



if __name__ == "__main__":

    for i in range(10):
        model = PPO(env_id = "InvertedDoublePendulum-v2", **inverted_double_pendulum_ppo)
        model.learn(total_timesteps = 1e+5, log_interval = 1, reward_target = 7000, log_to_file = False)


