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

from es import EvolutionStrategy

import gym
import pybullet_envs
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import vec_normalize
import pybulletgym
import mujoco_py

if __name__ == "__main__":

    for i in range(10):

        model = PPO(env_id = "BipedalWalker-v3", hidden_size = 64, lr = 0.001,
                                                max_grad_norm = 5, nstep = 256, batch_size = 128, n_epochs = 4, clip_range = 0.2, gamma = 0.99,
                                                gae_lam = 0.95)
        model.learn(1e+6, log_interval = 1, reward_target = 500, log_to_file = False)

        """
        model = EvolutionStrategy("Reacher-v2", [16,16], population_size = 20, sigma = 0.1, novelty_param = 1, learning_rate = 0.05, decay = 0.995, 
        num_threads = 4)
        model.run(1e+4, reward_target = 1000, log_to_file = False)
        """

    # InvertedPendulumSwingupPyBulletEnv-v0: 200
    # InvertedDoublePendulumPyBulletEnv-v0: 7000
    # ReacherPyBulletEnv-v0: 0
    # InvertedPendulumPyBulletEnv-v0: 900



