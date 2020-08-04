from gym import spaces
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from buffer import RolloutStorage, IntrinsicBuffer
from models import *
from util import RunningMeanStd
from agents import Agent

import time
from abc import ABC, abstractmethod

from collections import deque

from stable_baselines3.common import logger
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

class BaseAlgorithm(ABC):
    def __init__(self, env, 
                lr, 
                nstep, 
                batch_size, 
                n_epochs, 
                gamma, 
                gae_lam, 
                clip_range, 
                ent_coef, 
                vf_coef,
                max_grad_norm):   
        self.env = env
        self.lr = lr
        self.nstep = nstep
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lam = gae_lam
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.num_timesteps = 0 
        self.num_episodes = 0

        self.ep_info_buffer = deque(maxlen=100)
        self._n_updates = 0
        self.device = torch.device('cpu')
        
        self.num_envs = env.num_envs if isinstance(env, VecEnv) else 1

        self.action_space = env.action_space
        self.action_dim = self.action_space.n if self.action_space.__class__.__name__ == "Discrete" else self.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]

    @abstractmethod
    def collect_samples(self):
        """
        Collect rollouts using the current policy and fill a `RolloutBuffer`.

        """
        raise NotImplementedError()

    @abstractmethod
    def train(self):
        """
        Update policy using the currently gathered rollout buffer.

        """
        raise NotImplementedError()

    @abstractmethod
    def learn(self, total_timesteps, log_interval):
        """
        Initiate the learning process and return a trained model.

        """
        logger.configure('./logs')
        raise NotImplementedError()

    def update_info_buffer(self, infos, dones = None):
        """
        Retrieve reward and episode length and update the buffer
        if using Monitor wrapper.
        :param infos: ([dict])
        """
        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get('episode')
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])

class PPO(BaseAlgorithm):
    def __init__(self, *, 
                env, 
                lr = 3e-4, 
                nstep = 128, 
                batch_size = 128, 
                n_epochs = 4, 
                gamma = 0.99, 
                gae_lam = 0.9, 
                clip_range = 0.2, 
                ent_coef = .01, 
                vf_coef = 1,
                max_grad_norm = 0.2):   
        super(PPO, self).__init__(env, lr, nstep, batch_size, n_epochs, gamma, gae_lam, clip_range, ent_coef, vf_coef, max_grad_norm)                   
     

        self.policy = Policy(self.state_dim, self.action_dim)
        self.rollout = RolloutStorage(nstep, self.num_envs, env.observation_space, self.action_space, gae_lam = gae_lam)
        self.optimizer = optim.Adam(self.policy.parameters(), lr = lr)  

        self.last_obs = self.env.reset()

    def collect_samples(self):

        assert self.last_obs is not None
        
        rollout_step = 0
        self.rollout.reset()

        while rollout_step < self.nstep:
            with torch.no_grad():
                # Convert to pytorch tensor
                actions, values, log_probs = self.policy.act(self.last_obs)
            
            actions = actions.cpu().numpy()
            obs, rewards, dones, infos = self.env.step(actions)
            if any(dones):
                self.num_episodes += sum(dones)
            rollout_step += 1
            self.num_timesteps += self.num_envs
            self.update_info_buffer(infos)
            self.rollout.add(self.last_obs, actions, rewards, values, dones, log_probs)
            self.last_obs = obs

        self.rollout.compute_returns_and_advantages(values, dones=dones)

        return True

    def train(self):

        total_losses, policy_losses, value_losses, entropy_losses = [], [], [], []

        for epoch in range(self.n_epochs):
            for batch in self.rollout.get(self.batch_size):
                observations = batch.observations
                actions = batch.actions.long().flatten()
                old_log_probs = batch.old_log_probs
                old_values =    batch.old_values
                advantages =    batch.advantages
                returns =       batch.returns

                state_values, action_log_probs, entropy = self.policy.evaluate(observations, actions)

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(action_log_probs - old_log_probs)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                state_values_clipped = old_values + (state_values - old_values).clamp(-self.clip_range, self.clip_range)

                value_loss = F.mse_loss(returns, state_values).mean()
                value_loss_clipped = F.mse_loss(returns, state_values_clipped).mean()
                value_loss = torch.max(value_loss, value_loss_clipped).mean()

                entropy_loss = -torch.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_losses.append(loss.item())
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        logger.record("train/entropy_loss", np.mean(entropy_losses))
        logger.record("train/policy_gradient_loss", np.mean(policy_losses))
        logger.record("train/value_loss", np.mean(value_losses))
        logger.record("train/total_loss", np.mean(total_losses))

        self._n_updates += self.n_epochs
    
    def learn(self, total_timesteps, log_interval):
        start_time = time.time()
        iteration = 0

        while self.num_timesteps < total_timesteps:
            progress = round(self.num_timesteps/total_timesteps * 100, 2)
            self.collect_samples()
            
            iteration += 1
            if log_interval is not None and iteration % log_interval == 0:
                logger.record("Progress", str(progress)+'%')
                logger.record("time/total timesteps", self.num_timesteps)
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logger.record("rollout/ep_rew_mean",
                                  np.mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    logger.record("rollout/ep_len_mean",
                                  np.mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                    logger.record("rollout/num_episodes", self.num_episodes)
                fps = int(self.num_timesteps / (time.time() - start_time))
                logger.record("time/total_time", (time.time() - start_time))
                logger.dump(step=self.num_timesteps)   

            self.train()

        logger.record("Complete", '')
        logger.record("time/total timesteps", self.num_timesteps)
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            logger.record("rollout/ep_rew_mean",
                            np.mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            logger.record("rollout/ep_len_mean",
                            np.mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
            logger.record("rollout/num_episodes", self.num_episodes)
        fps = int(self.num_timesteps / (time.time() - start_time))
        logger.record("time/total_time", (time.time() - start_time))
        logger.dump(step=self.num_timesteps)   

        return self


class PPO_RND(PPO):
    def __init__(self, *, 
                env, 
                lr = 3e-4, 
                nstep = 128, 
                batch_size = 64, 
                n_epochs = 10, 
                gamma = 0.999, 
                int_gamma = 0.99,
                gae_lam = 0.95, 
                int_gae_lam = 0.95,
                clip_range = 0.2, 
                ent_coef = .01, 
                vf_coef = 0.5,
                int_vf_coef = 1.0,
                max_grad_norm = 0.5,
                rnd_start = 1e+3,
                hidden_size = 128):              
        super(PPO, self).__init__(env, lr, nstep, batch_size, n_epochs, gamma, gae_lam, clip_range, ent_coef, vf_coef, max_grad_norm) 
        
        
        self.int_gamma = int_gamma
        self.int_vf_coef = int_vf_coef
        self.int_gae_lam = int_gae_lam
        self.rnd_start = rnd_start

        self.policy = Policy(self.state_dim, self.action_dim, hidden_size = hidden_size)
        self.rnd = RndNetwork(env.observation_space.shape[0])
        self.rollout = IntrinsicBuffer(nstep, self.num_envs, env.observation_space, env.action_space, gae_lam = gae_lam, int_gae_lam = int_gae_lam)
        self.optimizer = optim.Adam(self.policy.parameters(), lr = lr)  
        self.rnd_optimizer = optim.Adam(self.rnd.parameters(), lr = lr)  

        self.last_obs = self.env.reset()

        self.requires_normalize = True


    def collect_samples(self):

        assert self.last_obs is not None
        
        rms_array = []

        rollout_step = 0
        self.rollout.reset()

        while rollout_step < self.nstep:
            with torch.no_grad():
                actions, ext_values, int_values, log_probs = self.policy.act(self.last_obs)
            
            actions = actions.numpy()
            obs, rewards, dones, infos = self.env.step(actions)
            if any(dones):
                self.num_episodes += sum(dones)
            self.num_timesteps += self.num_envs
            self.update_info_buffer(infos)

            if (self.num_timesteps / self.num_envs) < self.rnd_start:
                int_rewards = np.zeros_like(rewards)   

            else:
                int_rewards = self.rnd.int_reward(self.last_obs)

            self.rollout.add(self.last_obs, actions, rewards, int_rewards, ext_values, int_values, dones, log_probs)

            self.last_obs = obs
            rollout_step += 1

        self.rollout.compute_returns_and_advantages(ext_values, int_values, dones=dones)

        return True

    def train(self, rollout):
        """
        Update policy using the currently gathered rollout buffer.

        """
        total_losses, policy_losses, value_losses, entropy_losses, intrinsic_losses = [], [], [], [], []

        for epoch in range(self.n_epochs):
            for batch in rollout.get(self.batch_size):
                actions =           batch.actions.long().flatten()
                old_log_probs =     batch.old_log_probs
                ext_advantages =    batch.advantages
                int_advantages =    batch.int_advantages
                returns =           batch.returns
                int_returns =       batch.int_returns
                old_values =        batch.old_values
                old_int_values =    batch.int_values

                ext_values, int_values, action_log_probs, entropy = self.policy.evaluate(batch.observations, actions)
                ext_values = ext_values.squeeze()

                advantages = ext_advantages + int_advantages
                
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(action_log_probs - old_log_probs)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                ext_values_clipped = old_values + (ext_values - old_values).clamp(-self.clip_range, self.clip_range)
                ext_value_loss = F.mse_loss(returns, ext_values).mean()
                value_loss_clipped = F.mse_loss(returns, ext_values_clipped).mean()
                value_loss = torch.max(ext_value_loss, value_loss_clipped).mean()

                int_values_clipped = old_int_values + (int_values - old_int_values).clamp(-self.clip_range, self.clip_range)
                int_value_loss = F.mse_loss(int_returns, int_values).mean()
                int_value_loss_clipped = F.mse_loss(int_returns, int_values_clipped)
                intrinsic_loss = torch.max(int_value_loss, int_value_loss_clipped).mean()


                if entropy is None:
                    entropy_loss = -action_log_probs.mean()
                else:
                    entropy_loss = -torch.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.int_vf_coef * intrinsic_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_losses.append(loss.item())
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                intrinsic_losses.append(intrinsic_loss.item())

        logger.record("train/entropy_loss", np.mean(entropy_losses))
        logger.record("train/policy_gradient_loss", np.mean(policy_losses))
        logger.record("train/value_loss", np.mean(value_losses))
        logger.record("train/intrinsic_loss", np.mean(value_losses))
        logger.record("train/total_loss", np.mean(total_losses))

        self._n_updates += self.n_epochs

    def train_rnd(self, rollout):   
        for epoch in range(self.n_epochs):
            for batch in rollout.get(self.batch_size):
                obs = batch.observations #self.rew_norm_and_clip(batch.observations.numpy())
                pred, target = self.rnd(obs)

                loss = F.mse_loss(pred, target)

                self.rnd_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.rnd.parameters(), self.max_grad_norm)
                self.rnd_optimizer.step()

    
    def learn(self, total_timesteps, log_interval, n_eval_episodes = 5):
        start_time = time.time()
        iteration = 0

        while self.num_timesteps < total_timesteps:
            progress = round(self.num_timesteps/total_timesteps * 100, 2)
            self.collect_samples()
            
            iteration += 1
            if log_interval is not None and iteration % log_interval == 0:
                logger.record("Progress", str(progress)+'%')
                logger.record("time/total timesteps", self.num_timesteps)
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logger.record("rollout/ep_rew_mean",
                                  np.mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    logger.record("rollout/ep_len_mean",
                                  np.mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                    logger.record("rollout/num_episodes", self.num_episodes)
                fps = int(self.num_timesteps / (time.time() - start_time))
                logger.record("time/total_time", (time.time() - start_time))
                logger.dump(step=self.num_timesteps)  
 

            self.train(self.rollout)
            if np.random.randn() < 0.25:
                self.train_rnd(self.rollout)

        logger.record("Complete", '.')
        logger.record("time/total timesteps", self.num_timesteps)
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            logger.record("rollout/ep_rew_mean",
                            np.mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            logger.record("rollout/ep_len_mean",
                            np.mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        fps = int(self.num_timesteps / (time.time() - start_time))
        logger.record("time/total_time", (time.time() - start_time))
        logger.dump(step=self.num_timesteps)   

        return self

    def update_info_buffer(self, infos, dones = None):
        """
        Retrieve reward and episode length and update the buffer
        if using Monitor wrapper.
        :param infos: ([dict])
        """
        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get('episode')
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])

    def rew_norm_and_clip(self, obs):  
        norm_obs = (obs - self.rms_obs.mean/(np.sqrt(self.rms_obs.var) + 1e-8)).clip(-5, 5)
        return torch.FloatTensor(norm_obs.astype(float))



