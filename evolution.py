import logging
from models import *
import gym
import numpy as np
from agents import *
import itertools
import multiprocessing as mp

from collections import namedtuple
from queue import Queue
import copy

RewardsItem = namedtuple('RewardsItem', field_names=['seed', 'pos_reward', 'neg_reward', 'steps'])

ENV_ID = "MountainCar-v0"
iters_per_update = 10
lr = 1e-03
num_workers = 1
population_size = 50
SIGMA = 0.01


def evaluate(policy, env):
    obs = env.reset()
    total_r = 0
    total_steps = 0
    for t in itertools.count():
        actions, values, _ = policy.act(obs)
        obs, reward, done, info = env.step(actions.numpy())
        total_r += reward
        if done:
            total_steps = t
            break
    return total_r, total_steps

def evaluate_noisy(policy, noise, env):
    old_dict = copy.deepcopy(policy.net.state_dict())
    new_dict = dict()

    for n, key in zip(noise, policy.net.state_dict().keys()):
        new_dict[key] = policy.net.state_dict()[key] + (n * SIGMA)

    policy.net.load_state_dict(new_dict)  
    reward, steps = evaluate(policy, env)
    policy.net.load_state_dict(old_dict)
    
    return reward, steps

def sample_noise(policy):
    pos = list()
    neg = list()
    for param in policy.net.parameters():
        noise = np.random.normal(size = param.data.numpy().shape)
        pos.append(noise)
        neg.append(-noise)
    return pos, neg


def train_step(policy, batch_noise, batch_reward, step_idx):
    """
    Optimizes the weights of the NN based on the rewards and noise gathered
    """
    # normalize rewards to have zero mean and unit variance
        
    norm_reward = np.array(batch_reward)
    norm_reward = (norm_reward - np.mean(norm_reward)) / (np.std(norm_reward) + 1e-7)

    weighted_noise = None
    for noise, reward in zip(batch_noise, norm_reward):
        if weighted_noise is None:
            weighted_noise = [reward * p_n for p_n in noise]
        else:
            for w_n, p_n in zip(weighted_noise, noise):
                w_n += reward * p_n

    new_weights = dict()

    for key, p_update in zip(policy.net.state_dict().keys(), weighted_noise):
        update = p_update / (len(batch_reward)*SIGMA)
        new_weights[key] = policy.net.state_dict()[key] + lr * update
    
    policy.net.load_state_dict(new_weights)

def worker(worker_id, param_queue, rewards_queue):  
    env = gym.make(ENV_ID)
    agent = Policy(env = env)   
    agent.net.eval()  

    while not param_queue.empty():
        params = param_queue.get()  

        agent.net.load_state_dict(params)

        for _ in range(iters_per_update):
            # get a random seed
            seed = np.random.randint(1e6)
            # set the new seed
            np.random.seed(seed)
            pos_n, neg_n = np.array(sample_noise(agent))
            pos_rew, pos_steps = evaluate_noisy(agent, pos_n, env)
            neg_rew, neg_steps = evaluate_noisy(agent, neg_n, env)

            rewards_queue.put(RewardsItem(seed=seed, pos_reward=pos_rew, neg_reward=neg_rew, steps=pos_steps+neg_steps))           

    pass
        
def learn():
    iterations = 100
    
    params_queues = [mp.Queue(maxsize=1) for _ in range(4)]
    rewards_queue = mp.Queue(maxsize=iters_per_update)
    
    workers = []
            
    for idx, params_queue in enumerate(params_queues):
        proc = mp.Process(target=worker, args=(idx, params_queue, rewards_queue), daemon=True)
        proc.start()
        workers.append(proc)   

    step_idx = 0
    reward_history = []
    reward_max =[]
    reward_std = []
    
    for step_idx in range(iterations):
        # broadcasting network params
        policy = Policy(env = gym.make(ENV_ID))
        policy.net.eval()
        params = policy.net.state_dict()
        for q in params_queues:
            q.put(params)

        batch_noise = []
        batch_reward = []
        batch_steps_data = []
        batch_steps = 0
        results = 0
        while True: 
            print(rewards_queue.empty())
            while not rewards_queue.empty():
                reward = rewards_queue.get_nowait()
                np.random.seed(reward.seed) # sets the seed of the current worker rewards
                noise, neg_noise = sample_noise(brain)
                batch_noise.append(noise)
                batch_reward.append(reward.pos_reward)
                batch_noise.append(neg_noise)
                batch_reward.append(reward.neg_reward)
                results += 1
                batch_steps += reward.steps

            if results == num_workers * iters_per_update:
                break
                
            step_idx += 1
            m_reward = np.mean(batch_reward)
            reward_history.append(m_reward)
            reward_std.append(np.std(batch_reward))
            reward_max.append(np.max(batch_reward))
            if m_reward > 199:
                print("\nSolved the environment in {} steps".format(step_idx))
                break
                
            print(batch_reward)
            train_step(policy, batch_noise, batch_reward, step_idx)

            print("\rStep: {}, Mean_Reward: {:.2f}".format(step_idx, m_reward), end = "", flush = True)
            
        for w, p_queue in zip(workers, params_queues):
            p_queue.put(None)
            w.join()




        