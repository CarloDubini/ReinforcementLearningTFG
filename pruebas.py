import gymnasium as gym
import numpy as np
import matplotlib
from utils import plot_learning_curve
from Actor import Agent

env = gym.make('FetchReach-v2', max_episode_steps=100,render_mode="human")

observation = env.reset()

print(env.action_space.shape[0])