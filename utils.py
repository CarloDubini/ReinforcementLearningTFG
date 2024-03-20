from scipy.spatial import distance
import numpy as np
import math
import matplotlib.pyplot as plt
def plot_learning_curve(x, scores, figure_file, number=100):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-number):(i+1)])
    plt.plot(x, running_avg)
    plt.savefig(figure_file)
    plt.clf()

def plot_learning_curve_three(x, scores, figure_file, number=100):
    running_avg_global = np.zeros(len(scores))
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-number):(i+1)])
        running_avg_global[i] = np.mean(scores[max(0, i-len(scores)):(i+1)])
    plt.plot(x, running_avg_global, 'y')
    plt.plot(x, running_avg, 'g')
    plt.title('Y = mean global, B = Last 100 mean, K = scores')
    plt.savefig(figure_file)


def transformObservation(obs):
    observation = np.concatenate((obs['observation'], obs['desired_goal']),axis=None)
    return observation

def transformObservationHER(obs):
    observation = np.concatenate((obs['observation'], obs['achieved_goal']),axis=None)
    return observation


def euclidDistanceNegative(observation, goal):
    a = (observation[3], observation[4], observation[5])
    b = (goal[0], goal[1], goal[2])
    return -distance.euclidean(a, b)

def euclidDistanceNegativeTimesSquared(observation, goal):
    a = (observation[3], observation[4], observation[5])
    b = (goal[0], goal[1], goal[2])
    #Multiplicamos la distancia por 100 para medir la distancia en cm en vez de en metros. 
    #La razón de esta decisión es que cuando multiplicas dos numeros menores que uno el reward será menor.
    reward = math.pow(100 * distance.euclidean(a, b),2) 
    return -reward

def calcularRewardCuadratico(reward,cuadratico):
    if(cuadratico):
        return reward*reward
    else:
        return reward
    

def euclidDistanceNegativeCube(observation):
    return -math.sqrt(observation[6]**2+ observation[7]**2 + observation[8]**2)

def cubeReward(observation):
    o = transformObservation(observation)
    reward = euclidDistanceNegativeCube(o)
    return reward*0.5

