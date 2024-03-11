import gymnasium as gym
import numpy as np
import matplotlib
from utils import plot_learning_curve
from Actor import Actor

'''env = gym.make('FetchReach-v2', max_episode_steps=100,render_mode="human")
n_actions = env.action_space.shape[0]

obs = env.observation_space['observation'].sample()
numpyArray= np.concatenate((obs[0:3],env.observation_space['desired_goal'].sample()),axis=None)

env.set_goal()
'''
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
y = np.array([3,3,3])
observation_HER = np.concatenate((x[0:10], y), axis= 0)
print(observation_HER)


