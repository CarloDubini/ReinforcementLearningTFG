from Actor import Actor

def applyHER(agent, observation, action, new_observation, new_goal, done, goalmethod = None):
    if goalmethod is not None:
        reward = goalmethod(observation, new_goal)
        agent.remember(observation, action, reward, new_observation, done)