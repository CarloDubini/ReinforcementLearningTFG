import time
from scipy.spatial import distance
import gymnasium as gym
import numpy as np
from utils import *
from Actor import Actor
from HER import applyHER

def main():
    # Creación del entorno
    env = gym.make('FetchPushDense-v2',render_mode="human")
    n_actions = env.action_space.shape[0]

    # Creación del agente con el entorno y el número de acciones adecuados
    obs_array=env.observation_space.sample()
    numpyArray= transformObservation(obs_array)
 
    # Convert list to an array
    agent = Actor(input_dims=numpyArray.shape, environment=env, n_actions=n_actions, fc_dims= 350, alpha= 0.00001, beta= 0.00002, batch_size= 100, gamma= 0.99, noise= 0.002) 
    n_games = 1000 # Número de episodios a jugar
    max_iter = 100

    # Archivo para guardar la gráfica de rendimiento
    figure_file =  'plot/FetchPushPlot1.png'
    figure_file2 = 'plot/FetchPushPlot2.png'
    figure_file3 = 'plot/FetchPushPlot3.png'
    figure_file4 = 'plot/FetchPushPlot4.png'

    best_score = env.reward_range[0]  # Mejor puntuación inicializada con la peor posible
    score_history = []  # Lista para almacenar la puntuación en cada episodio
    cuadratic_negative = False #Flag para cambiar la recompensa cuadrática negativa
    continue_training = False # Flag con el objetivo de continuar entrenamientos 
    load_checkpoint = False  # Flag para cargar un punto de control previo
    train_with_HER = True # Aplicar HER durante el entrenamiento
    time_to_reward = True # Aplicar j a la recompensa para tener en cuenta la velocidad al objetivo.

    # Si se carga un punto de control, se inicializan las transiciones en el búfer de repetición
    if load_checkpoint or continue_training:
        
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
        if load_checkpoint:
            evaluate = True
        else:
            evaluate = False
    else:
        evaluate = False

   
    
    FinalScore = 0
    # Ciclo principal
    for i in range(n_games):

        observation = transformObservation(env.reset()[0])  # transformar el observation de un diccionario a un array 
            
        done = False
        score = 0
        j=0
        while not done and j<max_iter:
            action = agent.choose_action(observation, evaluate)  # Elegir una acción
            new_observation, reward, done, info, _ = env.step(action)  # Realizar la acción en el entorno
            
            reward = pushRewardDistanceNegative(new_observation['observation'], new_observation['desired_goal'])
            """
            if cuadratic_negative:
                reward = euclidDistanceNegativeTimesSquared(new_observation['observation'][3:6], new_observation['desired_goal'])
            """
            if time_to_reward and distance.euclidean(new_observation['observation'][3:6], new_observation['desired_goal']) > 0.1:
                reward += -j *0.01
            
            reward+= cubeReward(new_observation)

            score += reward  # Actualizar la puntuación acumulada  

            if train_with_HER:  
                new_goal = new_observation['achieved_goal']
                new_observation_HER = transformObservationHER(new_observation)
                observation_HER = np.concatenate((observation[0:25], new_goal), axis= 0)
                if cuadratic_negative:
                    applyHER(agent, observation_HER, action, new_observation_HER, new_goal, done)
                else:    
                    applyHER(agent, observation_HER, action, new_observation_HER, new_goal, done)
            
            new_observation = transformObservation(new_observation)

            agent.remember(observation, action, reward, new_observation, done)  # Almacenar la transición
            if not load_checkpoint:
                agent.learn()  # Aprender de la transición
            observation = new_observation  # Actualizar el estado actual
            
            j+=1
        
        
        if(n_games-i<=200):
            FinalScore= FinalScore+score
        
        score_history.append(score)  # Almacenar la puntuación del episodio
        avg_score = np.mean(score_history[-200:])  # Calcular la puntuación media en los últimos 100 episodios

        # Actualizar la mejor puntuación si se supera
        if avg_score > best_score:
            best_score = avg_score
            # Guardar los modelos del agente si no se cargó un punto de control previo
            if (not load_checkpoint) and (len(score_history)>200):
                 agent.save_models()

        # Imprimir información sobre el episodio actual
        print('episodio', i, 'puntuación %.1f' % score, 'puntuación media %.1f' % avg_score)


    print('Puntuacion Final:',FinalScore)
    # Graficar la curva de aprendizaje si no se cargó un punto de control previo

    first_training = np.array([])

    if continue_training:
        first_training = np.loadtxt("puntuacionesConHER.txt", delimiter= ",")
    
    score_history = np.array(score_history)

    score_history = np.concatenate((first_training, score_history), axis= 0)

    if not load_checkpoint:
        x = [i + 1 for i in range(score_history.shape[0])]
        plot_learning_curve(x, score_history, figure_file)
        plot_learning_curve(x, score_history, figure_file2 , n_games)
        plot_learning_curve(x, score_history, figure_file3 , 3)
        plot_learning_curve_three(x, score_history, figure_file4, n_games)
    
    env.close()

    np.savetxt("puntuacionesConHER.txt", score_history, delimiter= ",")
    
if __name__ == "__main__":
    main()
