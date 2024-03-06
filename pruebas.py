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
x = np.array([1,1,1])
y = np.array([3,3,3])
print(np.linalg.norm(x-y))



