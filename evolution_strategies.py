# Taken base code from https://github.com/alirezamika/evostra
# https://github.com/BY571/Deep-Reinforcement-Learning-Algorithm-Collection
import numpy as np
import multiprocessing as mp
import itertools
import time
import gym
import pybulletgym

from sklearn.neighbors import NearestNeighbors
from collections import deque
import logger

np.random.seed(0)

def worker_process(arg):
    get_reward_func, weights, env = arg
    return get_reward_func(weights, env)

def worker_process2(arg):
    get_reward_func, weights, archive = arg
    return get_reward_func(weights, archive)

import numpy as np

class FeedForwardNetwork(object):
    def __init__(self, env, hidden_sizes):
        self.env = env
        self.action_space = self.env.action_space.__class__.__name__
        self.weights = []

        layer_sizes = [env.observation_space.shape[0], *hidden_sizes, self.num_actions]
        for index in range(len(layer_sizes)-1):
            self.weights.append(np.random.randn(layer_sizes[index], layer_sizes[index+1]))


    @property
    def num_actions(self):
        if self.action_space == "Discrete":
            return self.env.action_space.n
        else:
            return self.env.action_space.shape[0]

    def predict(self, inp):
        out = np.expand_dims(inp.flatten(), 0)
        for i in range(len(self.weights)-1):
            try:
                out = np.dot(out, self.weights[i])
                out = np.arctan(out)
            except:
                continue

        out = np.dot(out, self.weights[-1])
        out = self.get_action(out.astype(float))
        return out

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def save(self, filename='weights.pkl'):
        with open(filename, 'wb') as fp:
            pickle.dump(self.weights, fp)

    def load(self, filename='weights.pkl'):
        with open(filename, 'rb') as fp:
            self.weights = pickle.load(fp)
    
    def softmax(self, logits):
        return np.exp(logits)/np.sum(np.exp(logits))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def discrete_action(self, n_actions, logits):
        return np.random.choice(np.arange(n_actions), p=self.softmax(logits).squeeze()).astype(int)

    def continuous_action(self, logits):
        mean = logits
        if self.env.observation_space.shape[0] == 1:
            return np.tanh(logits)
        else:
            return np.tanh(logits)

    def get_action(self, logits):
        if self.action_space == "Discrete":
            return self.discrete_action(self.env.action_space.n, logits)
        else:
            return self.continuous_action(logits).astype(np.double)


class EvolutionStrategy(object):
    def __init__(self, env_id, hidden_sizes, nsr_plateu = 1.5, nsr_range = [0, 1], nsr_update = 0.05, population_size=50, sigma=0.1, learning_rate=0.01, decay=0.9995, novelty_param = 0.5,
                 num_threads=1):
        self.env_id = env_id
        self.env = gym.make(env_id)
        self.hidden_sizes = hidden_sizes
        self.model = model = FeedForwardNetwork(self.env, hidden_sizes= hidden_sizes)
        self.weights = self.model.get_weights()
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.learning_rate = learning_rate
        self.decay = decay
        self.num_threads = mp.cpu_count() if num_threads == -1 else num_threads
        self.rewards = deque(maxlen=50)
        self.novelty_param = novelty_param
        self.K = 10
        self.nsr_plateu = nsr_plateu
        self.nsr_range = nsr_range
        self.nsr_update = nsr_update

    def _get_weights_try(self, w, p):
        weights_try = []
        for index, pop in enumerate(p):
            jittered = self.SIGMA * pop
            weights_try.append(w[index] + jittered)
        return weights_try

    def get_weights(self):
        return self.weights

    def evaluate(self, weights, env):
        self.model.set_weights(weights)
        obs = env.reset()
        total_r = 0
        total_steps = 0
        for t in itertools.count():
            actions = self.model.predict(obs)
            obs, reward, done, info = env.step(actions)
            total_r += reward
            if done:
                total_steps = t
                break
        return total_r

    def _get_population(self):
        population = []
        for i in range(self.POPULATION_SIZE):
            x = []
            for w in self.weights:
                x.append(np.random.randn(*w.shape))
            population.append(x)
        return population

    def _get_rewards(self, pool, population):
        if pool is not None:
            worker_args = ((self.evaluate, self._get_weights_try(self.weights, p), self.env) for p in population)
            rewards = pool.map(worker_process, worker_args)
        else:
            rewards = []
            for p in population:
                weights_try = self._get_weights_try(self.weights, p)
                rewards.append(self.evaluate(weights_try, self.env))
        rewards = np.array(rewards)
        return rewards.squeeze()

    def _get_rew_novelties(self, pool, population, archive):
        if pool is not None:
            worker_args = ((self.evaluate, self._get_weights_try(self.weights, p), self.env) for p in population)
            rewards = pool.map(worker_process, worker_args)

            novelty_args = ((self.get_novelty, self._get_weights_try(self.weights, p), archive) for p in population)
            novelties = pool.map(worker_process2, novelty_args)

        else:
            rewards = []
            novelties = []
            S = np.minimum(self.K, len(archive))
            for p in population:
                weights_try = self._get_weights_try(self.weights, p)
                rewards.append(self.evaluate(weights_try, self.env))
                b_phi_theta = self.get_behavior_char(weights_try, self.env)
                distance = self.get_kNN(archive, b_phi_theta, S)
                novelty = distance / S
                if novelty <= 1e-3:
                    novelty = 5e-3
                novelties.append(novelty)

        rewards = np.array(rewards)
        novelties = np.array(novelties)
        return rewards.squeeze(), novelties.squeeze()

    def get_novelty(self, p, archive):
        S = np.minimum(self.K, len(archive))
        b_phi_theta = self.get_behavior_char(p, self.env)
        distance = self.get_kNN(archive, b_phi_theta, S)
        novelty = distance / S
        if novelty <= 1e-3:
                novelty = 5e-3
        return novelty


    def _update_weights(self, rewards, population, novelty = None):
        std = rewards.std()
        if std == 0:
            return
        rewards = (rewards - rewards.mean()) / std
        for index, w in enumerate(self.weights):
            layer_population = np.array([p[index] for p in population])
            update_factor = self.learning_rate / (self.POPULATION_SIZE * self.SIGMA)
            if novelty is not None:
                #novelties = (novelties - np.mean(novelties)) / (np.std(novelties) + 1e-8)
                novelties = np.zeros(rewards.shape)
                novelties.fill(novelty)
                novelties_score = (1 - self.novelty_param) * np.dot(layer_population.T, rewards).T + self.novelty_param * np.dot(layer_population.T, novelties).T)/2
                self.weights[index] = w + update_factor * novelties_score
            else:
                self.weights[index] = w + update_factor * np.dot(layer_population.T, rewards).T         
        self.learning_rate *= self.decay

    def get_behavior_char(self, weights, env):
        """
        Returns the initial behavior characterization value b_pi0 for a network.
        The value is defined in this case as the final obs of agent in the environment.
    
        Important to find a good behavior characterization. Depents on the environment! <<< -> final obs, step count ... 
        
        """
        self.model.set_weights(weights)
        obs = env.reset()
        step_count = 0     
        while True:
            actions = self.model.predict(obs)
            obs, reward, done, info = env.step(actions)
            step_count += 1
            if done:
                final_pos = env.sim.data.qpos[0:2]
                break
        a = obs.reshape(1, self.env.observation_space.shape[0]) 
        return  np.array([final_pos])

    def get_kNN(self, archive, bc, n_neighbors):
        """
        Searches and samples the K-nearest-neighbors from the archive and a new behavior characterization
        returns the summed distance between input behavior characterization and the bc in the archive
        
        """
        archive = np.concatenate(archive)
        knn = NearestNeighbors(n_neighbors=n_neighbors)
        
        knn.fit(archive)
        distances, idx = knn.kneighbors(X = bc, n_neighbors=n_neighbors)
        #k_nearest_neighbors = archive[idx].squeeze(0)

        return sum(distances.squeeze(0))

    def calc_noveltiy_distribution(self, novelties):
        """
        Calculates the probabilities of each model parameters of being selected as its
        novelty normalized by the sum of novelty across all policies:
        P(theta_m) for each element in the meta_population M - m element M
        """
        probabilities = [round((novel/(sum(novelties))),4) for novel in novelties]
        return probabilities


    def run(self, total_timesteps, reward_target = None, log_interval=1, log_to_file = False):
        logger.configure("ES", self.env_id, log_to_file)

        MPS = 2
        meta_population = [FeedForwardNetwork(self.env, hidden_sizes= self.hidden_sizes) for _ in range(MPS)]

        pool = mp.Pool(self.num_threads) if self.num_threads > 1 else None
        start_time = time.time()

        archive = []
        delta_reward_buffer = deque(maxlen=10)
        
        novelties = []
        for iteration in range(int(total_timesteps)):
            population = self._get_population()   
            if len(archive) > 0:
                novelties = []
                S = np.minimum(self.K, len(archive))
                for model in meta_population:
                    b_pi_theta = self.get_behavior_char(model.get_weights(), self.env)
                    distance = self.get_kNN(archive, b_pi_theta, S)
                    novelty = distance / S
                    if novelty <= 1e-3:
                        novelty = 5e-3
                    novelties.append(novelty)

                probs = self.calc_noveltiy_distribution(novelties)
                
                probs = np.array(probs)
                probs /= probs.sum()   # norm so that sum up to one - does without as well but np gives error because of rounding
                brain_idx = np.random.choice(list(range(MPS)),p=probs) # select new brain based on novelty probabilities
                model = meta_population[brain_idx]
                novelty = novelties[brain_idx]

                self.model.set_weights(model.get_weights())
                rewards = self._get_rewards(pool, population)               
                self._update_weights(rewards, population, novelty) 
                meta_population[brain_idx].set_weights(self.model.get_weights())
            else:
                brain_idx = np.random.randint(0, MPS)
                model = meta_population[brain_idx]
                novelty = 1 

                self.model.set_weights(model.get_weights())
                rewards = self._get_rewards(pool, population)
                self._update_weights(rewards, population, novelty) 
                meta_population[brain_idx].set_weights(self.model.get_weights())
                
            mean_reward_batch = np.mean(rewards)
            reward_gradient_mean = np.mean(delta_reward_buffer)

            r_koeff = abs(mean_reward_batch - reward_gradient_mean)

            if iteration % 5 == 0:
                if r_koeff < self.nsr_plateu:
                    self.novelty_param = np.minimum(self.nsr_range[1], self.novelty_param + self.nsr_update)
                else:
                    self.novelty_param = np.maximum(self.nsr_range[0], self.novelty_param - self.nsr_update)

            delta_reward_buffer.append(mean_reward_batch)

                
            b_pix = self.get_behavior_char(self.weights, self.env)
            # append new behavior to specific brain archive
            archive.append(b_pix)
            
            self.rewards.extend([self.evaluate(self.weights, self.env)])

            if (iteration + 1) % log_interval == 0:
                logger.record("iteration", iteration + 1)
                logger.record("reward", np.mean(self.rewards))
                logger.record("novelty", np.mean(novelties))
                logger.record("n_koeff", self.novelty_param)
                logger.record("total_time", time.time() - start_time)
                logger.dump(step = iteration+1)
            if reward_target is not None and np.mean(self.rewards) > reward_target:
                print("Solved!")
                logger.record("iteration", iteration + 1)
                logger.record("reward", np.mean(self.rewards))
                logger.record("total_time", time.time() - start_time)
                logger.dump(step = iteration+1)             
                break
        if pool is not None:
            pool.close()
            pool.join()