﻿# PPO-exploration

Implementation of various exploration strategies for deep reinforcement learning.

Algorithms implemented:
* Proximal policy optomization (https://arxiv.org/abs/1707.06347)
* Similarity hashing with SimHash (https://arxiv.org/abs/1611.04717)
* Self supervised prediction with Intrinsic Curiosity Module(ICM) (https://arxiv.org/pdf/1705.05363.pdf)
* Random network distillation (RND) (https://arxiv.org/abs/1810.12894)
* Evolution Strategies based novelty search with Adaptive rewards (ES-NSRA) (https://arxiv.org/abs/1712.06560)

# Requirements

* Python3
* PyTorch
* Numpy
* Mujoco_py
* Gym
* stable_baselines3

# References for github repos used to aid creating this implementation

* https://github.com/DLR-RM/stable-baselines3
* https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
* https://github.com/adik993/ppo-pytorch
* https://github.com/openai/random-network-distillation
* https://github.com/topics/random-network-distillation
* https://github.com/BY571/Deep-Reinforcement-Learning-Algorithm-Collection
* https://github.com/alirezamika/evostra

# Performance
![Algorithm performance](images/time_comparison.png?raw=true "Title")
