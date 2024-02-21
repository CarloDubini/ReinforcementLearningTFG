import gymnasium as gym
import numpy as np
import matplotlib
from utils import plot_learning_curve
from Actor import Agent

env = gym.make('FetchReach-v2', max_episode_steps=100,render_mode="human")

observation = env.reset()

numpyArray= np.concatenate((env.observation_space['observation'].sample(),env.observation_space['desired_goal'].sample()),axis=None)

print(numpyArray.shape[0])
