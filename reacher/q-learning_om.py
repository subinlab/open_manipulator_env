#!/usr/bin/env python
import gym
import numpy as np 
import matplotlib.pyplot as plt 

env = gym.make('OMReach-v0')
env.render()

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Set learning parameters
learning_rate = 0.85
# Discount factor
dis = 0.99
num_episodes = 2000

# Create lists to contain total rewards and steps per episode
rList = []

for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False

    while not done:
        # Choose an action by greedily (with noise) piking form Q table
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i+1))

        # Get new state and reward form environment
        new_state, reward, done, _ = env.step(action)

        # Update Q function
        Q[state, action] = (1-learning_rate) * Q[state, action] + learning_rate * (reward + dis * np.max(Q[new_state, :]))

        rAll += reward
        state = new_state
    
    rList.append(rAll)

print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
