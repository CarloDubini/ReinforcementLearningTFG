from Actor import Actor

def applyHER(agent, observation, action, new_observation, new_goal, done, goalmethod = None):
    reward = 0
    if goalmethod is not None:
        reward = goalmethod(new_observation, new_goal)
    #print(observation)
    #print(reward)
    #print(new_observation)
    agent.remember(observation, action, reward, new_observation, done)