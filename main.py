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

if __name__ == "__main__":

    for i in range(10):
        model = EvolutionStrategy("Reacher-v2", [64,64], nsr_plateu = 1.5, nsr_range = [0,5], nsr_update = 0.05, sigma = 0.1, learning_rate = 0.03, decay = 0.995, 
                                    novelty_param = 1, num_threads = 4)
        model.run(total_timesteps = 1e+3, log_interval = 1, reward_target = -10, log_to_file = True)
    

    # ICM reacher hidden_size = 64, lr = 0.001, int_hidden_size = 16, int_lr = 0.0001, int_vf_coef = 0.05, nstep = 256
    # PPO(env_id = "InvertedDoublePendulum-v2", hidden_size = 64, lr = 0.001,  max_grad_norm = 5, nstep = 1024, batch_size = 128, n_epochs = 4, clip_range = 0.2,
    # PPO(env_id = "InvertedPendulum-v2", hidden_size = 64, lr = 0.001,  max_grad_norm = 5, nstep = 256, batch_size = 128, n_epochs = 4, clip_range = 0.2, gamma = 0.99,
    # PPO(env_id = "Reacher-v2", hidden_size = 64, lr = 0.001,  max_grad_norm = 5, nstep = 256, batch_size = 128, n_epochs = 4, clip_range = 0.2, gamma = 0.99,

    #PPO_RND(env_id = "Reacher-v2", hidden_size = 64, lr = 0.001, int_lr = 0.001, int_hidden_size = 32,  max_grad_norm = 5, nstep = 256, batch_size = 128, n_epochs = 4, clip_range = 0.2, gamma = 0.99,
    # PPO_RND(env_id = "InvertedPendulum-v2", hidden_size = 64, lr = 0.001, int_lr = 0.0001, int_hidden_size = 16,  max_grad_norm = 5, nstep = 256, batch_size = 128, n_epochs = 4, clip_range = 0.2, gamma = 0.99,
    # model = PPO_RND(env_id = "InvertedDoublePendulum-v2", hidden_size = 64, lr = 0.001, int_lr = 0.001, int_hidden_size = 32,  max_grad_norm = 5, nstep = 256, batch_size = 128, n_epochs = 4, clip_range = 0.2, gamma = 0.99,


    #PPO(env_id = "Swimmer-v2", hidden_size = 64, lr = 0.001, gamma = 0.995, gae_lam = 0.98, max_grad_norm = 5, 
    #                nstep = 2048, batch_size = 128, n_epochs = 10, clip_range = 0.2, ent_coef = 0.0)

    #PPO(env_id = "BipedalWalker-v3", hidden_size = 64, lr = 0.0003, gamma = 0.99, gae_lam = 0.95, max_grad_norm = 5, 
              #      nstep = 256, batch_size = 64, n_epochs = 4, clip_range = 0.2, ent_coef = 0.0)

    # PPO_RND(env_id = "InvertedDoublePendulum-v2", hidden_size = 64, lr = 0.00025, gamma = 0.99, gae_lam = 0.95, max_grad_norm = 5, rnd_start = 1e+3, 
    #                nstep = 2048, batch_size = 128, n_epochs = 10, clip_range = 0.2, ent_coef = 0.00, int_vf_coef = 0.5, int_hidden_size = 32, int_lr = 0.001)
