from gym import spaces
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from buffer import RolloutStorage, IntrinsicStorage
from models import *
from env import make_env
from util import RunningMeanStd, ActionConverter

import time
from abc import ABC, abstractmethod

from collections import deque
import multiprocessing

import logger
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

class BaseAlgorithm(ABC):
    """
    Base algorithm class that each agent has to inherit from.
    :param env_id: (str)            name of environment to perform training on
    :param lr: (float)              learning rate
    :param nstep: (int)             storage rollout steps
    :param batch_size: (int)        batch size for training
    :param n_epochs: (int)          number of training epochs
    :param gamma: (float)           discount factor
    :param gae_lam: (float)         lambda for generalized advantage estimation
    :param clip_range: (float)      clip range for surrogate loss
    :param ent_coef: (float)        entropy loss coefficient
    :param vf_coef: (float)         value loss coefficient
    :param max_grad_norm: (float)   max grad norm for optimizer
    """
    def __init__(self, 
                env_id, 
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

        self.env_id = env_id
        try:
            self.env = make_env(env_id, n_envs = 4)
        except:
            self.env = make_env(env_id, n_envs = 1, vec_env_cls = DummyVecEnv) 
            
        self.num_envs = self.env.num_envs if isinstance(self.env, VecEnv) else 1
        self.state_dim = self.env.observation_space.shape[0]
        self.action_converter = ActionConverter(self.env.action_space)

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

        self.ep_info_buffer = deque(maxlen=50)
        self._n_updates = 0
        self.num_timesteps = 0 
        self.num_episodes = 0

        
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
                env_id, 
                lr = 3e-4, 
                nstep = 128, 
                batch_size = 128, 
                n_epochs = 10, 
                gamma = 0.99, 
                gae_lam = 0.95, 
                clip_range = 0.2, 
                ent_coef = .01, 
                vf_coef = 1,
                max_grad_norm = 0.2,
                hidden_size = 128):   
        super(PPO, self).__init__(env_id, lr, nstep, batch_size, n_epochs, gamma, gae_lam, clip_range, ent_coef, vf_coef, max_grad_norm)                   

        self.policy = Policy(self.env, hidden_size)
        self.rollout = RolloutStorage(nstep, self.num_envs, self.env.observation_space, self.env.action_space, gae_lam = gae_lam, gamma = gamma)
        self.optimizer = optim.Adam(self.policy.net.parameters(), lr = lr, eps = 1e-5)  

        self.last_obs = self.env.reset()

    def collect_samples(self):
        assert self.last_obs is not None    
        rollout_step = 0
        self.rollout.reset()

        while rollout_step < self.nstep:
            with torch.no_grad():
                actions, values, log_probs = self.policy.act(self.last_obs)
            
            actions = actions.numpy() 
            obs, rewards, dones, infos = self.env.step(actions)
            if any(dones):
                self.num_episodes += sum(dones)

            self.num_timesteps += self.num_envs
            self.update_info_buffer(infos)

            actions = actions.reshape(self.num_envs, self.action_converter.action_output)
            log_probs = log_probs.reshape(self.num_envs, self.action_converter.action_output)
            
            self.rollout.add(self.last_obs, actions, rewards, values, dones, log_probs)
            self.last_obs = obs
            rollout_step += 1

        self.rollout.compute_returns_and_advantages(values, dones=dones)

        return True

    def train(self):
        total_losses, policy_losses, value_losses, entropy_losses = [], [], [], []

        for epoch in range(self.n_epochs):
            for batch in self.rollout.get(self.batch_size):
                observations =  batch.observations
                actions =       batch.actions
                old_log_probs = batch.old_log_probs
                old_values =    batch.old_values
                advantages =    batch.advantages
                returns =       batch.returns

                # Get values and action probabilities using the updated policy on gathered observations
                state_values, action_log_probs, entropy = self.policy.evaluate(observations, actions)
                
                # Normalize batch advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Compute policy gradient ratio of current actions probs over previous
                ratio = torch.exp(action_log_probs - old_log_probs)

                # Compute surrogate loss
                surr_loss_1 = advantages * ratio
                surr_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(surr_loss_1, surr_loss_2).mean()

                # Clip state values for stability
                state_values_clipped = old_values + (state_values - old_values).clamp(-self.clip_range, self.clip_range)
                value_loss = F.mse_loss(returns, state_values).mean()
                value_loss_clipped = F.mse_loss(returns, state_values_clipped).mean()
                value_loss = torch.max(value_loss, value_loss_clipped).mean()

                # Compute entropy loss
                entropy_loss = -torch.mean(entropy)

                # Total loss
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Perform optimization
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.net.parameters(), self.max_grad_norm)
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
    
    def learn(self, total_timesteps, log_interval, reward_target = None, log_to_file = False):
        logger.configure("PPO", self.env_id, log_to_file)
        start_time = time.time()
        iteration = 0

        while self.num_timesteps < total_timesteps:
            self.collect_samples()
            
            iteration += 1
            if log_interval is not None and iteration % log_interval == 0:
                logger.record("time/total timesteps", self.num_timesteps)
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logger.record("rollout/ep_rew_mean",
                                  np.mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    logger.record("rollout/num_episodes", self.num_episodes)
                fps = int(self.num_timesteps / (time.time() - start_time))
                logger.record("time/total_time", (time.time() - start_time))
                logger.dump(step=self.num_timesteps)   

            self.train()

            if reward_target is not None and np.mean([ep_info["r"] for ep_info in self.ep_info_buffer]) > reward_target:
                logger.record("time/total timesteps", self.num_timesteps)
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logger.record("rollout/ep_rew_mean",
                                    np.mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    logger.record("rollout/num_episodes", self.num_episodes)
                fps = int(self.num_timesteps / (time.time() - start_time))
                logger.record("time/total_time", (time.time() - start_time))
                logger.dump(step=self.num_timesteps)   

                break

        return self

class PPO_RND(BaseAlgorithm):
    def __init__(self, *, 
                env_id, 
                lr = 3e-4, 
                nstep = 128, 
                batch_size = 128, 
                n_epochs = 10, 
                gamma = 0.99, 
                gae_lam = 0.95, 
                clip_range = 0.2, 
                ent_coef = .01, 
                vf_coef = 1,
                int_vf_coef = 0.5,
                max_grad_norm = 0.2,
                hidden_size = 128,
                int_hidden_size = 128, 
                int_lr = 3e-4,
                rnd_start = 1e+4):   
        super(PPO_RND, self).__init__(env_id, lr, nstep, batch_size, n_epochs, gamma, gae_lam, clip_range, ent_coef, vf_coef, max_grad_norm)                   

        self.policy = Policy(self.env, hidden_size, intrinsic_model = True)
        self.rnd = RndNetwork(self.state_dim, hidden_size = int_hidden_size)
        self.rollout = IntrinsicStorage(nstep, self.num_envs, self.env.observation_space, self.env.action_space, gae_lam = gae_lam, gamma = gamma)
        self.optimizer = optim.Adam(self.policy.net.parameters(), lr = lr)  
        self.rnd_optimizer = optim.Adam(self.rnd.parameters(), lr = int_lr)

        self.rnd_start = rnd_start
        self.int_vf_coef = int_vf_coef

        self.last_obs = self.env.reset()
        self.rnd_normalizer = RunningMeanStd()   
        self.normalize = True
        self.last_dones = np.array([0 for _ in range(self.num_envs)])

    def collect_samples(self):
        assert self.last_obs is not None    
        rollout_step = 0
        self.rollout.reset()

        while rollout_step < self.nstep:
            with torch.no_grad():
                actions, values, int_values, log_probs = self.policy.act(self.last_obs)
            
            actions = actions.numpy() 
            obs, rewards, dones, infos = self.env.step(actions)
            if any(dones):
                self.num_episodes += sum(dones)

            self.num_timesteps += self.num_envs
            self.update_info_buffer(infos)

            actions = actions.reshape(self.num_envs, self.action_converter.action_output)
            log_probs = log_probs.reshape(self.num_envs, self.action_converter.action_output)

            if (self.num_timesteps / self.num_envs) < self.rnd_start:
                int_rewards = np.zeros_like(rewards)   
                """
                unnormalized_obs = self.env.unnormalize_obs(self.last_obs)
                mean, std, count = np.mean(unnormalized_obs), np.std(unnormalized_obs), len(unnormalized_obs)
                self.rnd_normalizer.update_from_moments(mean, std ** 2, count)
                """
            else:
                #normalized_obs = self.normalize_observations(self.last_obs)
                int_rewards = self.rnd.int_reward(self.last_obs).detach().numpy()
                
                
            self.rollout.add(self.last_obs, actions, rewards, int_rewards, values, int_values, dones, log_probs)
            self.last_obs = obs
            self.last_dones = dones
            rollout_step += 1

        self.rollout.compute_returns_and_advantages(values, int_values, dones)

        return True

    def train(self):
        total_losses, policy_losses, value_losses, entropy_losses, intrinsic_losses = [], [], [], [], []

        for epoch in range(self.n_epochs):
            for batch in self.rollout.get(self.batch_size):
                observations =  batch.observations
                actions =       batch.actions
                old_log_probs = batch.old_log_probs
                old_values =    batch.old_values
                old_int_values= batch.int_values
                advantages =    batch.advantages
                int_advantages= batch.int_advantages
                returns =       batch.returns
                int_returns =   batch.int_returns

                # Get values and action probabilities using the updated policy on gathered observations
                state_values, int_values, action_log_probs, entropy = self.policy.evaluate(observations, actions)
                
                # Normalize batch advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                int_advantages = (int_advantages - int_advantages.mean()) / (int_advantages.std() + 1e-8)

                advantages = advantages + int_advantages

                # Compute policy gradient ratio of current actions probs over previous
                ratio = torch.exp(action_log_probs - old_log_probs)

                # Compute surrogate loss
                surr_loss_1 = advantages * ratio
                surr_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(surr_loss_1, surr_loss_2).mean()

                # Clip state values for stability
                state_values_clipped = old_values + (state_values - old_values).clamp(-self.clip_range, self.clip_range)
                value_loss = F.mse_loss(returns, state_values).mean()
                value_loss_clipped = F.mse_loss(returns, state_values_clipped).mean()
                value_loss = torch.max(value_loss, value_loss_clipped).mean()

                # Clip state values for stability
                int_values_clipped = old_int_values + (int_values - old_int_values).clamp(-self.clip_range, self.clip_range)
                int_value_loss = F.mse_loss(int_returns, int_values).mean()
                int_value_loss_clipped = F.mse_loss(int_returns, int_values_clipped).mean()
                int_value_loss = torch.max(int_value_loss, int_value_loss_clipped).mean()

                # Compute entropy loss
                entropy_loss = -torch.mean(entropy)

                # Total loss
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.int_vf_coef * int_value_loss

                # Perform optimization
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_losses.append(loss.item())
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                intrinsic_losses.append(int_value_loss.item())

        logger.record("train/intrinsic_loss", np.mean(intrinsic_losses))
        logger.record("train/entropy_loss", np.mean(entropy_losses))
        logger.record("train/policy_gradient_loss", np.mean(policy_losses))
        logger.record("train/value_loss", np.mean(value_losses))
        logger.record("train/total_loss", np.mean(total_losses))   

        self._n_updates += self.n_epochs

    def train_rnd(self):   
        for epoch in range(self.n_epochs):
            for batch in self.rollout.get(self.batch_size):
                obs = batch.observations #self.rew_norm_and_clip(batch.observations.numpy())
                pred, target = self.rnd(obs)

                loss = F.mse_loss(pred, target)

                self.rnd_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.rnd.parameters(), self.max_grad_norm)
                self.rnd_optimizer.step()
    
    def learn(self, total_timesteps, log_interval, reward_target = None, log_to_file = False):
        logger.configure("RND", self.env_id, log_to_file)
        start_time = time.time()
        iteration = 0

        while self.num_timesteps < total_timesteps:
            self.collect_samples()
            
            iteration += 1
            if log_interval is not None and iteration % log_interval == 0:
                logger.record("time/total timesteps", self.num_timesteps)
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logger.record("rollout/ep_rew_mean",
                                  np.mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    logger.record("rollout/num_episodes", self.num_episodes)
                fps = int(self.num_timesteps / (time.time() - start_time))
                logger.record("time/total_time", (time.time() - start_time))
                logger.dump(step=self.num_timesteps)   

            self.train()
            if np.random.randn() < 0.25:
                self.train_rnd()

            if reward_target is not None and np.mean([ep_info["r"] for ep_info in self.ep_info_buffer]) > reward_target:
                logger.record("time/total timesteps", self.num_timesteps)
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logger.record("rollout/ep_rew_mean",
                                    np.mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    logger.record("rollout/num_episodes", self.num_episodes)
                fps = int(self.num_timesteps / (time.time() - start_time))
                logger.record("time/total_time", (time.time() - start_time))
                logger.dump(step=self.num_timesteps)   

                break

        return self
    def normalize_observations(self, obs):
        return np.clip(((obs - self.rnd_normalizer.mean) / np.sqrt(self.rnd_normalizer.var)), -5, 5)


class PPO_ICM(BaseAlgorithm):
    def __init__(self, *, 
                env_id, 
                lr = 3e-4, 
                int_lr = 3e-4,
                nstep = 128, 
                batch_size = 128, 
                n_epochs = 10, 
                gamma = 0.99, 
                gae_lam = 0.95, 
                clip_range = 0.2, 
                ent_coef = .01, 
                vf_coef = 1,
                int_vf_coef = 0.1,
                max_grad_norm = 0.2,
                hidden_size = 128,
                int_hidden_size = 32):   
        super(PPO_ICM, self).__init__(env_id, lr, nstep, batch_size, n_epochs, gamma, gae_lam, clip_range, ent_coef, vf_coef, max_grad_norm)                   
     
        self.int_vf_coef = int_vf_coef

        self.policy = Policy(self.env, hidden_size)
        self.rollout = RolloutStorage(nstep, self.num_envs, self.env.observation_space, self.env.action_space, gae_lam = gae_lam)
        self.intrinsic_module = IntrinsicCuriosityModule(self.state_dim, self.action_converter, hidden_size = int_hidden_size)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr = lr)  
        self.icm_optimizer = optim.Adam(self.intrinsic_module.parameters(), lr = int_lr)

        self.last_obs = self.env.reset()
        self.int_norm = RunningMeanStd()    

    def collect_samples(self):

        assert self.last_obs is not None
        
        rollout_step = 0
        self.rollout.reset()

        # For logging
        test_int_rewards = []


        while rollout_step < self.nstep:
            with torch.no_grad():
                # Convert to pytorch tensor
                actions, values, log_probs = self.policy.act(self.last_obs)

            obs, rewards, dones, infos = self.env.step(actions.numpy())

            if any(dones):
                self.num_episodes += sum(dones)
            rollout_step += 1
            self.num_timesteps += self.num_envs
            self.update_info_buffer(infos)

            
            int_rewards = self.intrinsic_module.int_reward(torch.Tensor(self.last_obs), torch.Tensor(obs), actions)

            # For logging
            test_int_rewards.append(int_rewards.mean().item())

            rewards = (1-self.int_vf_coef) * rewards + self.int_vf_coef * int_rewards.detach().numpy()
            actions = actions.reshape(self.num_envs, self.action_converter.action_output)
            log_probs = log_probs.reshape(self.num_envs, self.action_converter.action_output)

            self.rollout.add(self.last_obs, 
                            actions, 
                            rewards, 
                            values, 
                            dones, 
                            log_probs)

            self.last_obs = obs
        logger.record("rollout/mean_int_reward", np.round(np.mean(np.array(test_int_rewards)), 2))
        self.rollout.compute_returns_and_advantages(values, dones=dones)

        return True

    def train(self):
        total_losses, policy_losses, value_losses, entropy_losses, icm_losses = [], [], [], [], []

        inv_criterion = self.action_converter.get_loss()

        for epoch in range(self.n_epochs):
            for batch in self.rollout.get(self.batch_size):
                observations =  batch.observations
                actions =       batch.actions
                old_log_probs = batch.old_log_probs
                old_values =    batch.old_values
                advantages =    batch.advantages
                returns =       batch.returns

                state_values, action_log_probs, entropy = self.policy.evaluate(observations, actions)

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = torch.exp(action_log_probs - old_log_probs)

                # Surrogate loss
                surr_loss_1 = advantages * ratio
                surr_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(surr_loss_1, surr_loss_2).mean()

                # Clipped value loss
                state_values_clipped = old_values + (state_values - old_values).clamp(-self.clip_range, self.clip_range)
                value_loss = F.mse_loss(returns, state_values).mean()
                value_loss_clipped = F.mse_loss(returns, state_values_clipped).mean()
                value_loss = torch.max(value_loss, value_loss_clipped).mean()

                # Icm loss

                actions_hat, next_features, next_features_hat = self.intrinsic_module(observations[:-1], observations[1:], actions[:-1])

                forward_loss = F.mse_loss(next_features, next_features_hat)
                inverse_loss = inv_criterion(actions_hat, self.action_converter.action(actions[:-1]))
                icm_loss = inverse_loss + forward_loss

                entropy_loss = -torch.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + icm_loss

                self.optimizer.zero_grad()
                self.icm_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.icm_optimizer.step()

                total_losses.append(loss.item())
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                icm_losses.append(icm_loss.item())

        logger.record("train/entropy_loss", np.mean(entropy_losses))
        logger.record("train/policy_gradient_loss", np.mean(policy_losses))
        logger.record("train/value_loss", np.mean(value_losses))
        logger.record("train/total_loss", np.mean(total_losses))
        logger.record("train/icm_loss", np.mean(icm_losses))

        self._n_updates += self.n_epochs
    
    def learn(self, total_timesteps, log_interval = 5, reward_target = None, log_to_file = False):
        logger.configure("ICM", self.env_id, log_to_file)
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
                    logger.record("rollout/num_episodes", self.num_episodes)
                fps = int(self.num_timesteps / (time.time() - start_time))
                logger.record("time/total_time", (time.time() - start_time))
                logger.dump(step=self.num_timesteps)   

            self.train()

            if reward_target is not None and np.mean([ep_info["r"] for ep_info in self.ep_info_buffer]) > reward_target:
                logger.record("time/total timesteps", self.num_timesteps)
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    logger.record("rollout/ep_rew_mean",
                                    np.mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    logger.record("rollout/num_episodes", self.num_episodes)
                fps = int(self.num_timesteps / (time.time() - start_time))
                logger.record("time/total_time", (time.time() - start_time))
                logger.dump(step=self.num_timesteps)   
                break

        return self