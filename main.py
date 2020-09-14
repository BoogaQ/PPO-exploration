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

        model = EvolutionStrategy("InvertedDoublePendulum-v2", [32,32], nsr_plateu = 10, nsr_range = [0,0.5], nsr_update = 0.05, sigma = 0.1, learning_rate = 0.03, decay = 0.9995, 
                                    novelty_param = 0, num_threads = 4)
        model.run(total_timesteps = 1e+5, log_interval = 1, reward_target = 7000, log_to_file = False)


