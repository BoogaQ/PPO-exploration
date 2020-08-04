from gym import spaces
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from buffer import RolloutStorage, IntrinsicBuffer
from models import *
from util import RunningMeanStd

import time
from abc import ABC, abstractmethod

from collections import deque

from stable_baselines3.common import logger
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

class BaseAlgorithm(ABC):
    def __init__(self, env, 
                learning_rate, 
                buffer_size, 
                batch_size, 
                n_epochs, 
                gamma, 
                gae_lam, 
                clip_range, 
                ent_coef, 
                vf_coef,
                max_grad_norm):   
        self.env = env
        self.lr = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lam = gae_lam
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.num_timesteps = 0 

        self.ep_info_buffer = deque(maxlen=100)
        self._n_updates = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if isinstance(env, VecEnv):
            self.num_envs = env.num_envs 

        self.rms_obs = RunningMeanStd(shape = (1,1,84,84))
        self.rms_rew = RunningMeanStd()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.configure('./logs')

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
                learning_rate = 3e-4, 
                buffer_size = 128, 
                batch_size = 64, 
                n_epochs = 10, 
                gamma = 0.99, 
                gae_lam = 0.95, 
                clip_range = 0.1, 
                ent_coef = .01, 
                vf_coef = 1.0,
                max_grad_norm = 0.5):   
        super(PPO, self).__init__(env, learning_rate, buffer_size, batch_size, n_epochs, gamma, gae_lam, clip_range, ent_coef, vf_coef, max_grad_norm)                   
     

        self.policy = Policy(env = env, device = self.device)
        self.rollout = RolloutStorage(buffer_size, self.num_envs, env.observation_space, env.action_space, gae_lam = gae_lam)
        self.optimizer = optim.Adam(self.policy.parameters(), lr = learning_rate)  

        self.last_obs = self.env.reset()


    def collect_samples(self):

        assert self.last_obs is not None
        
        rollout_step = 0
        self.rollout.reset()

        while rollout_step < self.buffer_size:
            with torch.no_grad():
                # Convert to pytorch tensor
                actions, values, log_probs = self.policy.act(self.last_obs)
            
            actions = actions.cpu()
            obs, rewards, dones, infos = self.env.step(actions)
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
                actions = batch.actions.long().flatten()
                old_log_probs = batch.old_log_probs.to(self.device)
                advantages =    batch.advantages.to(self.device)
                returns =       batch.returns.to(self.device)

                state_values, action_log_probs, entropy = self.policy.evaluate(batch.observations, actions)
                state_values = state_values.squeeze()

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(action_log_probs - old_log_probs)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                value_loss = F.mse_loss(returns, state_values)

                if entropy is None:
                    entropy_loss = -action_log_probs.mean()
                else:
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
                fps = int(self.num_timesteps / (time.time() - start_time))
                logger.record("time/total_time", (time.time() - start_time))
                logger.dump(step=self.num_timesteps)   

            self.train()

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


class PPO_RND(PPO):
    def __init__(self, *, 
                env, 
                learning_rate = 3e-4, 
                buffer_size = 128, 
                batch_size = 64, 
                n_epochs = 10, 
                gamma = 0.999, 
                int_gamma = 0.99,
                gae_lam = 0.95, 
                int_gae_lam = 0.95,
                clip_range = 0.1, 
                ent_coef = .01, 
                vf_coef = 1.0,
                max_grad_norm = 0.5,
                rnd_start = 1e+3):              
        super(PPO, self).__init__(env, learning_rate, buffer_size, batch_size, n_epochs, gamma, gae_lam, clip_range, ent_coef, vf_coef, max_grad_norm) 
        
        
        self.int_gamma = int_gamma
        self.int_gae_lam = int_gae_lam
        self.rnd_start = rnd_start

        self.policy = IntrinsicActorCritic(4, env.action_space.n).to(self.device)
        self.rnd = RndNetwork().to(self.device)
        self.rollout = IntrinsicBuffer(buffer_size, self.num_envs, env.observation_space, env.action_space, gae_lam = gae_lam, int_gae_lam = int_gae_lam)
        self.optimizer = optim.Adam(self.policy.parameters(), lr = learning_rate)  
        self.rnd_optimizer = optim.Adam(self.rnd.parameters(), lr = learning_rate)  

        self.last_obs = self.env.reset()

        self.requires_normalize = True


    def collect_samples(self):

        assert self.last_obs is not None
        
        rms_array = []

        rollout_step = 0
        self.rollout.reset()

        while rollout_step < self.buffer_size:
            with torch.no_grad():
                actions, ext_values, int_values, log_probs = self.policy.act(self.last_obs)
            
            actions = actions.cpu()
            obs, rewards, dones, infos = self.env.step(actions)
            self.num_timesteps += self.num_envs
            self.update_info_buffer(infos)

            if (self.num_timesteps / self.num_envs) < self.rnd_start:
                """for i in range(len(self.last_obs)):
                    rms_array.append(self.last_obs[i, -1])

                if rollout_step == self.buffer_size - 1:
                    rms_array = np.stack(rms_array)
                    self.rms_obs.update(rms_array)"""

                int_rewards = np.zeros_like(rewards)   

            else:
                #norm_obs = self.rew_norm_and_clip(self.last_obs).to(self.device)
                last_frames = torch.FloatTensor(self.last_obs[:,None,3,:,:]).to(self.device)
                int_rewards = self.rnd.int_reward(last_frames)

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
                old_log_probs =     batch.old_log_probs.to(self.device)
                ext_advantages =    batch.advantages.to(self.device)
                int_advantages =    batch.int_advantages.to(self.device)
                returns =           batch.returns.to(self.device)
                int_returns =       batch.int_returns.to(self.device)

                ext_values, int_values, action_log_probs, entropy = self.policy.evaluate(batch.observations, actions)
                ext_values = ext_values.squeeze()

                advantages = ext_advantages + int_advantages
                
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(action_log_probs - old_log_probs)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                value_loss = F.mse_loss(returns, ext_values)
                intrinsic_loss = F.mse_loss(int_returns, int_values)

                if entropy is None:
                    entropy_loss = -action_log_probs.mean()
                else:
                    entropy_loss = -torch.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + intrinsic_loss

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
                obs = batch.observations.to(self.device) #self.rew_norm_and_clip(batch.observations.numpy()).to(self.device)
                pred, target = self.rnd(obs[:,None,3,:,:])

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



