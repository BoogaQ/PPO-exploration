from util import *
from buffer import *

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class SilModule():
    def __init__(self, size, policy, optimizer, n_envs, env, gamma = .99):
        self.policy = policy
        self.trajectories = [[] for _ in range(n_envs)]
        self.optimizer = optimizer
        self.buffer = PrioritizedReplayBuffer(size, 1, n_envs, env.observation_space, env.action_space)

        # some other parameters...
        self.gamma = gamma
        self.total_steps = []
        self.total_rewards = []

    def step(self, obs, actions, log_probs, rewards, dones):
        for n in range(len(self.trajectories)):
            self.trajectories[n].append([obs[n], actions[n], log_probs[n], rewards[n]])

        for n, done in enumerate(dones):
            if done:
                self.update_buffer(self.trajectories[n])
                self.trajectories[n] = []

    def update_buffer(self, trajectory):
        self.add_episode(trajectory)
        self.total_steps.append(len(trajectory))
        self.total_rewards.append(np.sum([x[3] for x in trajectory]))
        while np.sum(self.total_steps) > self.buffer.buffer_size and len(self.total_steps) > 1:
            self.total_steps.pop(0)
            self.total_rewards.pop(0)

    def add_episode(self, trajectory):
        obs = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        for (ob, action, log_prob, reward) in trajectory:
            obs.append(ob)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(False)
        dones[len(dones) - 1] = True
        returns = self.discount_with_dones(rewards, dones)
        for (ob, action, log_prob, R) in list(zip(obs, actions, log_probs, returns)):
            self.buffer.add(ob, action, log_prob, R)

    def train(self, n_epochs, batch_size, clip_range, ent_coef = 0.01):
        sil_losses, mean_advantages = [], []
        for epoch in range(n_epochs):
            if self.sample_batch(batch_size) is not None:
                observations, actions, old_log_probs, returns, indices = next(self.sample_batch(batch_size))
                observations = observations.float()

                state_values, action_log_probs, entropy = self.policy.evaluate(observations, actions)

                ratio = torch.exp(action_log_probs - old_log_probs)

                advantages = returns - state_values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                clipped_advantages = torch.clamp(advantages, 0, 10).unsqueeze(1)

                masks = (advantages.detach().numpy() > 0).astype(np.float32)
                masks = torch.tensor(masks, dtype=torch.float32).unsqueeze(1)         

                # Compute surrogate loss
                surr_loss_1 = clipped_advantages * ratio
                surr_loss_2 = clipped_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(surr_loss_1, surr_loss_2).mean()

                entropy_loss = -(entropy * masks).mean()
                policy_loss = policy_loss + ent_coef * entropy_loss

                value_loss = (clipped_advantages).mean()
                total_loss = 0.1 * policy_loss + 0.01 * value_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.net.parameters(), 1)
                self.optimizer.step()

                self.buffer.update_priorities(indices, clipped_advantages.detach().numpy())

                sil_losses.append(total_loss.item())
                mean_advantages.append(clipped_advantages.mean().item())

        logger.record("rollout/mean_sil_advantage", np.mean(mean_advantages))  
        logger.record("train/sil_loss", np.mean(sil_losses))       

    def discount_with_dones(self, rewards, dones):
        discounted = []
        r = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + self.gamma * r * (1. - done)
            discounted.append(r)
        return discounted[::-1]


    def sample_batch(self, batch_size):
        if len(self.buffer) > 100:
            batch_size = min(batch_size, len(self.buffer))
            return self.buffer.get(batch_size)
        else:
            return None

