import numpy as np
import torch

from collections import namedtuple



class BaseBuffer(object):
    """
    Base class that represent a buffer (rollout or replay)
    :param buffer_size: (int) Max number of element in the buffer
    :param observation_space: (spaces.Space) Observation space
    :param action_space: (spaces.Space) Action space
    :param device: (Union[th.device, str]) Pynp device
        to which the values will be converted
    :param n_envs: (int) Number of parallel environments
    """

    def __init__(self,
                 buffer_size,
                 observation_space,
                 action_space,
                 device = 'cpu',
                 n_envs = 1):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = observation_space.shape
        self.action_dim = action_space.n
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr):
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)
        :param arr: (np.ndarray)
        :return: (np.ndarray)
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self):
        """
        :return: (int) The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs):
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def reset(self):
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self,
               batch_size: int,
               env = None
               ):
        """
        :param batch_size: (int) Number of element to sample
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: (Union[RolloutBufferSamples, ReplayBufferSamples])
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds, env = None):
        """
        :param batch_inds: (th.Tensor)
        :param env: (Optional[VecNormalize])
        :return: (Union[RolloutBufferSamples, ReplayBufferSamples])
        """
        raise NotImplementedError()

    def to_torch(self, array, copy = True):
        """
        Convert a numpy array to a Pynp tensor.
        Note: it copies the data by default
        :param array: (np.ndarray)
        :param copy: (bool) Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return: (th.Tensor)
        """
        if copy:
            return torch.tensor(array).to(self.device)
        return torch.as_tensor(array).to(self.device)

    @staticmethod
    def _normalize_obs(obs,
                       env = None):
        if env is not None:
            return env.normalize_obs(obs).astype(np.float32)
        return obs

    @staticmethod
    def _normalize_reward(reward,
                          env = None):
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward



class RolloutStorage(BaseBuffer):
    def __init__(self, buffer_size, n_envs, obs_space, action_space, gae_lam = 0.95, gamma = 0.99):
        super(RolloutStorage, self).__init__(buffer_size, obs_space, action_space, n_envs = n_envs)

        self.gae_lam = gae_lam
        self.gamma = gamma

        self.observations, self.actions, self.rewards, self.values = None, None, None, None
        self.returns, self.action_log_probs, self.masks, self.advantages = None, None, None, None  
        self.generator_ready = False

        self.step = 0
        self.full = False
        self.reset()

        self.RolloutSample = namedtuple('RolloutSample', ['observations', 'actions', 'old_values', 'old_log_probs', 'advantages', 'returns'])
        

    def reset(self):
        self.observations =     np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype = 'float32')
        self.actions =          np.zeros((self.buffer_size, self.n_envs), dtype = 'long')
        self.rewards =          np.zeros((self.buffer_size, self.n_envs), dtype = 'float32')
        self.values =           np.zeros((self.buffer_size, self.n_envs), dtype = 'float32')
        self.returns =          np.zeros((self.buffer_size, self.n_envs), dtype = 'float32')
        self.action_log_probs = np.zeros((self.buffer_size, self.n_envs), dtype = 'float32')
        self.masks =            np.ones((self.buffer_size, self.n_envs), dtype = 'long')
        self.advantages =       np.zeros((self.buffer_size, self.n_envs), dtype = 'float32')
        self.generator_ready = False
        super(RolloutStorage, self).reset()   

    def add(self, obs, action, reward, value, mask, log_prob):
        """
        :param obs: (np.Tensor) Observation
        :param action: (np.Tensor) Action
        :param reward: (np.Tensor)
        :param done: (np.Tensor) End of episode signal.
        :param value: (np.Tensor) estimated value of the current state following the current policy.
        :param log_prob: (np.Tensor) log probability of the action following the current policy.
        """

        self.observations[self.pos] =       np.array(obs).copy()
        self.actions[self.pos] =            action.clone().cpu().numpy()
        self.rewards[self.pos] =            np.array(reward).copy()
        self.masks[self.pos] =              np.array(mask).copy()
        self.values[self.pos] =             value.clone().cpu().numpy().flatten()
        self.action_log_probs[self.pos] =   log_prob.clone().cpu().numpy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True


    def compute_returns_and_advantages(self, last_value, dones):
        """
        Post-processing step: compute the returns (sum of discounted rewards)
        and GAE advantage.
        Adapted from Stable-Baselines PPO2.
        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lam=1.0`` during initialization.

        :param last_value: (th.Tensor)
        :param dones: (np.ndarray)
        """

        last_value = last_value.clone().cpu().numpy().flatten()
        last_gae_lam = 0

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.masks[step + 1]
                next_value = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lam * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def get(self, batch_size = None):
        assert self.full, ''
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        if not self.generator_ready:
            for tensor in ['observations', 'actions', 'values',
                           'action_log_probs', 'advantages', 'returns']:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx:start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds, env = None):
        data = (self.observations[batch_inds],
                self.actions[batch_inds],
                self.values[batch_inds].flatten(),
                self.action_log_probs[batch_inds].flatten(),
                self.advantages[batch_inds].flatten(),
                self.returns[batch_inds].flatten())
        return self.RolloutSample(*tuple(map(self.to_torch, data)))
