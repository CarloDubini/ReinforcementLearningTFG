import time
import gymnasium as gym
import numpy as np
from utils import euclidDistanceNegative, plot_learning_curve, plot_learning_curve_three,transformObservation
from Actor import Actor

def main():
    # Creación del entorno
    env = gym.make('FetchReachDense-v2',render_mode="human")
    n_actions = env.action_space.shape[0]

    # Creación del agente con el entorno y el número de acciones adecuados
    obs_array=env.observation_space.sample()
    numpyArray= transformObservation(obs_array)
 
    # Convert list to an array
    agent = Actor(input_dims=numpyArray.shape, environment=env, n_actions=n_actions, fc_dims= 100, alpha= 0.00001, beta= 0.00002, noise= 0.03,tau=0.01) 
    n_games = 2  # Número de episodios a jugar
    max_iter = 50

    # Archivo para guardar la gráfica de rendimiento
    figure_file =  'plot/FetchReachPlot1.png'
    figure_file2 = 'plot/FetchReachPlot2.png'
    figure_file3 = 'plot/FetchReachPlot3.png'
    figure_file4 = 'plot/FetchReachPlot4.png'

    best_score = env.reward_range[0]  # Mejor puntuación inicializada con la peor posible
    score_history = []  # Lista para almacenar la puntuación en cada episodio
    load_checkpoint = False  # Bandera para cargar un punto de control previo

    # Si se carga un punto de control, se inicializan las transiciones en el búfer de repetición
    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_len:
            observation = env.reset()
            action = env.action_space.sample()
            # observation = np.concatenate((env.observation_space['observation'].sample(), env.observation_space['desired_goal'].sample()),axis=None)
            observation_, reward, done, info, _ = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            n_steps += 1
        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False

   
    goal = [0,0,0]
    goal_set = False
    # Ciclo principal
    for i in range(n_games):

        observation = transformObservation(env.reset()[0])  # transformar el observation de un diccionario a un array 
            
        done = False
        score = 0
        j=0
        while not done and j<max_iter:
            action = agent.choose_action(observation, evaluate)  # Elegir una acción
            observation_, reward, done, info, _ = env.step(action)  # Realizar la acción en el entorno
            if goal_set == False:
                goal = observation_["desired_goal"]
                goal_set = True
            observation_= transformObservation(observation_)

            reward = euclidDistanceNegative(observation, goal)
            if(reward>-0.2):
                reward= reward*0.5
            
            if j < 25 and j > 23: 
                print(reward)

            score += reward  # Actualizar la puntuación acumulada
            agent.remember(observation, action, reward, observation_[0], done)  # Almacenar la transición
            if not load_checkpoint:
                agent.learn()  # Aprender de la transición
            observation = observation_  # Actualizar el estado actual
            
            j+=1

        score_history.append(score)  # Almacenar la puntuación del episodio
        avg_score = np.mean(score_history[-200:])  # Calcular la puntuación media en los últimos 100 episodios

        # Actualizar la mejor puntuación si se supera
        if avg_score > best_score:
            best_score = avg_score
            # Guardar los modelos del agente si no se cargó un punto de control previo
            if not load_checkpoint:
                 agent.save_models()

        # Imprimir información sobre el episodio actual
        print('episodio', i, 'puntuación %.1f' % score, 'puntuación media %.1f' % avg_score)


    # Graficar la curva de aprendizaje si no se cargó un punto de control previo
    if not load_checkpoint:
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
        plot_learning_curve(x, score_history, figure_file2 , n_games)
        plot_learning_curve(x, score_history, figure_file3 , 3)
        plot_learning_curve_three(x, score_history, figure_file4, n_games)
    
    env.close()
    
if __name__ == "__main__":
    main()
