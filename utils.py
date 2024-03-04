import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file, number=100):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-number):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous scores')
    plt.savefig(figure_file)


def transformObservation(obs):
    observation = np.concatenate((obs['observation'][0:3],obs['desired_goal']),axis=None)
    return observation