import numpy as np
import torch.nn as nn
import torch

class RunningMeanStd(object):
    def __init__(self, epsilon = 1e-4, shape = ()):
        """
        Calulates the running mean and std of a data stream. Adopted for multiple parallel environments.
        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
        print(np.mean(self.mean), np.mean(self.var), self.count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class ActionConverter():
    def __init__(self, action_space):
        self.action_type = action_space.__class__.__name__
        if self.action_type == "Discrete":
            self.num_actions = action_space.n
            self.action_output = 1
        elif self.action_type == "Box":
            self.num_actions = action_space.shape[0]
            self.action_output = self.num_actions

    def get_loss(self):
        if self.action_type == "Discrete":
            loss = nn.CrossEntropyLoss()
        elif self.action_type == "Box":
            loss = nn.MSELoss()
        return loss

    def action(self, action):
        if self.action_type == "Discrete":
            return action.long()
        elif self.action_type == "Box":
            return action.float()
            

