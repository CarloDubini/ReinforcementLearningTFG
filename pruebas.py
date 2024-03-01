import gymnasium as gym
import numpy as np
import matplotlib
from utils import plot_learning_curve
from Actor import Actor

env = gym.make('FetchReach-v2', max_episode_steps=100,render_mode="human")
n_actions = env.action_space.shape[0]
observation = env.reset()[0]
print(observation['observation'][0:3])
obs = env.observation_space['observation'].sample()
numpyArray= np.concatenate((obs[0:3],env.observation_space['desired_goal'].sample()),axis=None)

print(n_actions)
