import time
from scipy.spatial import distance
import gymnasium as gym
import random as rnd
import numpy as np
import sys
from utils import calcularRewardCuadratico, euclidDistanceNegative, plot_learning_curve, plot_learning_curve_three,transformObservation, euclidDistanceNegativeTimesSquared, transformObservationHER
from Actor import Actor
from HER import applyHER

def main():
    # Creación del entorno
    env = gym.make('FetchReachDense-v2',render_mode="human")
    n_actions = env.action_space.shape[0]

    # Creación del agente con el entorno y el número de acciones adecuados
    obs_array=env.observation_space.sample()
    numpyArray= transformObservation(obs_array)
    #Lista de hiperparámetros:

    n_games = 1000  # Número de episodios a jugar
    her_statistic = 0.8
    max_iter = 50
    dim_layers = [250, 150, 50]
    alpha = 0.0001
    beta = 0.0005
    batch_size= 50
    gamma= 0.99
    noise= 0.001
    
    agent = Actor(input_dims=numpyArray.shape, environment=env, n_actions=n_actions, dim_layers= dim_layers, 
                  alpha= alpha, beta= beta, batch_size= batch_size, gamma= gamma, noise= noise) 

    # Archivo para guardar la gráfica de rendimiento
    figure_file =  f'plot/FetchReachPlot1HERSPARSa{alpha}b{beta}bs{dim_layers}noise{noise}.png'
    figure_file2 = f'plot/FetchReachPlot2HERSPARSa{alpha}b{beta}{dim_layers}noise{noise}.png'
    figure_file3 = f'plot/FetchReachPlot3HERSPARSEa{alpha}b{beta}{dim_layers}noise{noise}.png'
    figure_file4 = f'plot/FetchReachPlot4HERSPARSa{alpha}b{beta}{dim_layers}noise{noise}.png'

    best_score = sys.float_info.min  # Mejor puntuación inicializada con la peor posible
    score_history = []  # Lista para almacenar la puntuación en cada episodio
    cuadratic_negative = False #Flag para cambiar a la recompensa cuadrática negativa
    continue_training = False # Flag con el objetivo de continuar entrenamientos 
    explotation_mode = False  # Flag para cargar un punto de control previo y explotar el modelo
    train_with_HER = True # Aplicar HER durante el entrenamiento
    time_to_reward = True # Añadir a la recompensa una 

    # Si se carga un punto de control, se inicializan las transiciones en el búfer de repetición
    if explotation_mode or continue_training:
        
        n_steps = 0
        while n_steps <= agent.batch_len:
            observation = transformObservation(env.reset()[0])
            action = env.action_space.sample()
            new_observation, reward, done, info, _ = env.step(action)
            new_observation= transformObservation(new_observation)
            agent.remember(observation, action, reward, new_observation, done)
            n_steps += 1
        agent.learn()
        agent.load_models()
        if explotation_mode:
            evaluate_training = True
        else:
            evaluate_training = False
    else:
        evaluate_training = False

   
    
    FinalScore = 0
    # Ciclo principal
    for i in range(n_games):

        observation = transformObservation(env.reset()[0])  # transformar el observation de un diccionario a un array 
            
        done = False
        score = 0
        j=0
        while not done and j<max_iter:
            action = agent.choose_action(observation, evaluate_training)  # Elegir una acción con o sin ruido según si se está evaluando o no
            new_observation, reward, done, info, _ = env.step(action)  # Realizar la acción en el entorno

            if cuadratic_negative:
                reward = euclidDistanceNegativeTimesSquared(new_observation['observation'][0:3], new_observation['desired_goal'])
            
            if time_to_reward and distance.euclidean(new_observation['observation'][0:3], new_observation['desired_goal']) > 0.1:
                reward += -j *0.01
            
            score += reward  # Actualizar la puntuación acumulada  

            if train_with_HER and rnd.random() < her_statistic:  
                new_goal = new_observation['achieved_goal']
                new_observation_HER = transformObservationHER(new_observation)
                observation_HER = np.concatenate((observation[0:10], new_goal), axis= 0)
                if cuadratic_negative:
                    applyHER(agent, observation_HER, action, new_observation_HER, new_goal, done,  euclidDistanceNegativeTimesSquared)
                else:    
                    applyHER(agent, observation_HER, action, new_observation_HER, new_goal, done)
            
            new_observation = transformObservation(new_observation)
            agent.remember(observation, action, reward, new_observation, done)  # Almacenar la transición
            if not explotation_mode:
                agent.learn()  # Aprender de la transición
            observation = new_observation  # Actualizar el estado actual
            
            j+=1
        
        
        if(n_games-i<=200):#
            FinalScore= FinalScore+score
        
        score_history.append(score)  # Almacenar la puntuación del episodio
        avg_score = np.mean(score_history[-200:])  # Calcular la puntuación media en los últimos 100 episodios

        # Actualizar la mejor puntuación si se supera
        if avg_score > best_score and i > n_games/4:
            best_score = avg_score
            # Guardar los modelos del agente si no se cargó un punto de control previo
            if (not explotation_mode) and (len(score_history)>200):
                 agent.save_models()

        # Imprimir información sobre el episodio actual
        print('episodio', i, 'puntuación %.1f' % score, 'puntuación media %.1f' % avg_score)


    print('Puntuacion Final:',FinalScore)
    # Graficar la curva de aprendizaje si no se cargó un punto de control previo

    first_training = np.array([])

    if continue_training:
        first_training = np.loadtxt(f"puntuacionesconHERa{alpha}b{beta}{dim_layers}noise{noise}.txt", delimiter= ",")
    
    score_history = np.array(score_history)

    score_history = np.concatenate((first_training, score_history), axis= 0)

    if not explotation_mode:
        x = [i + 1 for i in range(score_history.shape[0])]
        plot_learning_curve(x, score_history, figure_file)
        plot_learning_curve(x, score_history, figure_file2 , n_games)
        plot_learning_curve(x, score_history, figure_file3 , 3)
        plot_learning_curve_three(x, score_history, figure_file4, n_games)
    
    env.close()

    np.savetxt(f"punt{alpha}b{beta}bs{dim_layers}noise{noise}.txt", score_history, delimiter= ",")
    
if __name__ == "__main__":
    main()
