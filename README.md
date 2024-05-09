# Application of Advanced Reinforcement Learning Algorithms for Continuous Control

## Authors:
- Pablo Pardos Medem
- Carlo Sebastiano Dubini Marqués
## Abstract:
In current times, automatic control is being introduced increasingly in the industry and our lives, becoming something quite common to be the object of study through machine learning techniques, even more, to perform complex tasks in continuous space, highlighting its application in robotics and automatic control. This paper discusses the different characteristics of reinforcement learning, neural networks, and training of these techniques, highlighting their importance in machine learning and their practical applications. Subsequently, Policy Gradient Methods are explained in detail, responsible for dealing with problems in continuous space without a real need for discretization, emphasizing DDPG and HER within these. Not only these are leading and advanced techniques within reinforcement learning, but they have been implemented to execute the control of a pendulum and an industrial robot arm.  With the data following the training of this robot arm in MuJoCo, we have found the parameters that are considered most important for this by iterating on possible changes to the structures, optimizers, and control parameters to perform a complete analysis to add additional goals and obstacle considerations for future work that can be iterated on these hypotheticals, explaining the steps to follow for its correct implementation.

Dependencies:
 - matplotlib
 - tensorflow
 - gym
 - gym[classic_control]
 - gymnasium.robotics
 - mujoco
 - keras
