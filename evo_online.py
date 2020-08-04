import numpy as np
import matplotlib.pyplot as plt
import gym
import multiprocessing as mp
import collections
import copy

from models import Policy
import itertools

ITERS_PER_UPDATE = 10
NOISE_STD = 0.01
LR = 5e-3
PROCESSES_COUNT = 4 # amount of worker
HIDDEN_SIZE = 4
ENV_NAME = "CartPole-v0" 
RewardsItem = collections.namedtuple('RewardsItem', field_names=['seed', 'pos_reward', 'neg_reward', 'steps'])

        
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

def sample_noise(policy):
    """
    Sampling noise to a positive and negative noise buffer.
    """
    pos = []
    neg = []
    for param in policy.parameters():
        noise_t = np.random.normal(size = param.shape)
        pos.append(noise_t)
        neg.append(-noise_t)
    return pos, neg


def eval_with_noise(env, policy, noise, noise_std):
    """
    Evaluates the current brain with added parameter noise
  
    """

    old_dict = copy.deepcopy(policy.net.state_dict())
    new_dict = dict()

    for n, key in zip(noise, policy.net.state_dict().keys()):
        new_dict[key] = policy.net.state_dict()[key] + (n * NOISE_STD)

    policy.net.load_state_dict(new_dict)  
    reward, steps = evaluate(policy, env)
    policy.net.load_state_dict(old_dict)
    
    return reward, steps


def worker_func(worker_id, params_queue, rewards_queue, noise_std):
    #print("worker: {} has started".format(worker_id))
    env = gym.make(ENV_NAME)
    net = Policy(env = env)
    while True:
        params = params_queue.get()
        if params is None:
            break

        # set parameters of the queue - equal to: net.load_state_dict(params)
        net.net.load_state_dict(params)
        
        for _ in range(ITERS_PER_UPDATE):
            seed = np.random.randint(low=0, high=65535)
            np.random.seed(seed)
            noise, neg_noise = sample_noise(net)
            pos_reward, pos_steps = eval_with_noise(env, net, noise, noise_std)
            neg_reward, neg_steps = eval_with_noise(env, net, neg_noise, noise_std)
            #print(_, "\n",noise, pos_reward, neg_reward)
            
            rewards_queue.put(RewardsItem(seed=seed, pos_reward=pos_reward, neg_reward=neg_reward, steps=pos_steps+neg_steps))

    pass


def train_step(policy, batch_noise, batch_rewards, step_idx):
    """
    Optimizes the weights of the NN based on the rewards and noise gathered
    """
    # normalize rewards to have zero mean and unit variance
    norm_reward = np.array(batch_reward)
    norm_reward -= np.mean(norm_reward)
    s = np.std(norm_reward)
    if abs(s) > 1e-6:
        norm_reward /= s
        
    weighted_noise = None
    for noise, reward in zip(batch_noise, norm_reward):
        if weighted_noise is None:
            weighted_noise = [reward * p_n for p_n in noise]
        else:
            for w_n, p_n in zip(weighted_noise, noise):
                w_n += reward * p_n
        
    new_weights = dict()

    for key, p_update in zip(policy.net.state_dict().keys(), weighted_noise):
        update = p_update / (len(batch_reward)*NOISE_STD)
        new_weights[key] = policy.net.state_dict()[key] + LR * update
    
    policy.net.load_state_dict(new_weights)
        
        
if __name__ == "__main__":

    env = gym.make(ENV_NAME)
    #env.seed(2)
    policy = Policy(env = env)

    iterations = 100 # max iterations to run 

    params_queues = [mp.Queue(maxsize=1) for _ in range(PROCESSES_COUNT)]
    rewards_queue = mp.Queue(maxsize=ITERS_PER_UPDATE)
    
    
    workers = []


    for idx, params_queue in enumerate(params_queues):
        proc = mp.Process(target=worker_func, args=(idx, params_queue, rewards_queue, NOISE_STD))
        proc.start()
        workers.append(proc)

    print("All started!")
    step_idx = 0
    reward_history = []
    reward_max =[]
    reward_std = []


    for step_idx in range(iterations):
        # broadcasting network params
        params = policy.net.state_dict()
        for q in params_queues:
            q.put(params)

        batch_noise = []
        batch_reward = []
        batch_steps_data = []
        batch_steps = 0
        results = 0
        while True: 
            while not rewards_queue.empty():
                reward = rewards_queue.get_nowait()
                np.random.seed(reward.seed) # sets the seed of the current worker rewards
                noise, neg_noise = sample_noise(policy)
                batch_noise.append(noise)
                batch_reward.append(reward.pos_reward)
                batch_noise.append(neg_noise)
                batch_reward.append(reward.neg_reward)
                results += 1
                batch_steps += reward.steps

            if results == PROCESSES_COUNT * ITERS_PER_UPDATE:
                break

        step_idx += 1
        m_reward = np.mean(batch_reward)
        reward_history.append(m_reward)
        reward_max.append(np.max(batch_reward))
        reward_std.append(np.std(batch_reward))
        if m_reward > 199:
            print("\nSolved the environment in {} steps".format(step_idx))
            break
        train_step(policy, batch_noise, batch_reward, step_idx)

        print("\rStep: {}, Mean_Reward: {:.2f}".format(step_idx, m_reward), end = "", flush = True)


    for worker, p_queue in zip(workers, params_queues):
        p_queue.put(None)
        worker.join()

    plt.figure(figsize = (11,7))
    plt.plot(reward_history, label = "Mean Reward", color = "orange")
    plt.plot(reward_max, label = "Max Reward", color = "blue")
    plt.plot(reward_std, label = "Reward std", color = "green")
    plt.xlabel("Steps")
    plt.ylabel("Rewards")
    plt.legend()
    plt.show()