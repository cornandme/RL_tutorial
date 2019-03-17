import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

### Set learning parameters ###

# iterations
num_episodes = 2000

# Discount factor
dis = .99

# learning rate
learning_rate = .85

# choose Explore method
# 1: with noise / 2: e-greedy
exp_method = 1

# stochastic world
env = gym.make('FrozenLake-v0')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Create lists to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False

    # The Q-Table learning algorithm
    while not done:
        # Choose an action by greedily picking from Q table
        # 1. with noise
        if exp_method == 1:
            action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i+1))
        # 2. e-greedy
        elif exp_method == 2:
            # e-greedy factor
            e = 1. / ((i // (100))+1)
            if np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        # Update Q-Table with new knowledge using decay rate
        Q[state,action] = ((1-learning_rate) * Q[state,action]) + (learning_rate * (reward + dis * np.max(Q[new_state, :])))

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Score over time: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()