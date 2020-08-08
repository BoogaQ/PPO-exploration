# Taken base code from https://github.com/alirezamika/evostra
import numpy as np
import multiprocessing as mp
import itertools
import time

from sklearn.neighbors import NearestNeighbors

np.random.seed(0)


def worker_process(arg):
    get_reward_func, weights, env = arg
    return get_reward_func(weights, env)


class EvolutionStrategy(object):
    def __init__(self, model, env, population_size=50, sigma=0.1, learning_rate=0.03, decay=0.999,
                 num_threads=1):

        self.model = model
        self.weights = self.model.get_weights()
        self.env = env
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.learning_rate = learning_rate
        self.decay = decay
        self.num_threads = mp.cpu_count() if num_threads == -1 else num_threads

    def _get_weights_try(self, w, p):
        weights_try = []
        for index, i in enumerate(p):
            jittered = self.SIGMA * i
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

    def _update_weights(self, rewards, population):
        std = rewards.std()
        if std == 0:
            return
        rewards = (rewards - rewards.mean()) / std
        for index, w in enumerate(self.weights):
            layer_population = np.array([p[index] for p in population])
            update_factor = self.learning_rate / (self.POPULATION_SIZE * self.SIGMA)
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
                break
        return  np.array([obs]) #obs 

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

    def get_novelties(self)

    def run(self, iterations, print_step=1):
        pool = mp.Pool(self.num_threads) if self.num_threads > 1 else None
        start_time = time.time()
        archive = []
        for iteration in range(iterations):
        
            population = self._get_population()
            rewards = self._get_rewards(pool, population)

            



            self._update_weights(rewards, population)

            if (iteration + 1) % print_step == 0:
                print('iter %d. reward: %f. total_time: %f.' % (iteration + 1, self.evaluate(self.weights, self.env), time.time() - start_time))
        if pool is not None:
            pool.close()
            pool.join()