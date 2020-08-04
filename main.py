import multiprocessing as mp
from evolution import *
from queue import Queue


if __name__ == '__main__':
    iterations = 100
    env = gym.make(ENV_ID)
    policy = Actor(env.observation_space.shape[0], env.action_space.n, HIDDEN_SIZE) 
        
    params_queues = Queue(maxsize=1)
    rewards_queue = Queue(maxsize=iters_per_update)

    step_idx = 0
    reward_history = []
    reward_max =[]
    reward_std = []
    
    for step_idx in range(iterations):      
        params = policy.state_dict()
        params_queues.put(params)

        worker(0, params_queues, rewards_queue)

        batch_noise = []
        batch_reward = []
        batch_steps_data = []
        batch_steps = 0
        results = 0

        while True: 
            while not rewards_queue.empty():
                reward = rewards_queue.get()
                np.random.seed(reward.seed) # sets the seed of the current worker rewards
                pos_noise, neg_noise = np.array(sample_noise(policy))
                batch_noise.append(pos_noise)
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

        train_step(policy, batch_noise, batch_reward, step_idx)
        
        print("\rStep: {}, Mean_Reward: {:.2f}".format(step_idx, m_reward), end = "", flush = True)
            