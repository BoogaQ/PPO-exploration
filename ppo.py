from gym import spaces
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from buffer import RolloutStorage
from models import Policy

import time

from collections import deque

from stable_baselines3.common import logger

class PPO():
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

        if env.num_envs:
            self.num_envs = env.num_envs 

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

        self.policy = Policy(env = env, device = self.device)
        self.rollout = RolloutStorage(buffer_size, self.num_envs, env.observation_space, env.action_space, gae_lambda = gae_lam)
        self.optimizer = optim.Adam(self.policy.parameters(), lr = learning_rate)  

        self.last_obs = self.env.reset()

        logger.configure('./logs')


    def collect_rollouts(self):
        """
        Collect rollouts using the current policy and fill a `RolloutBuffer`.

        """
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
        """
        Update policy using the currently gathered rollout buffer.

        """
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
    
    def learn(self, total_timesteps, log_interval, eval_freq = -1, n_eval_episodes = 5):
        start_time = time.time()
        iteration = 0

        while self.num_timesteps < total_timesteps:
            progress = round(self.num_timesteps/total_timesteps * 100, 2)
            self.collect_rollouts()
            
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
