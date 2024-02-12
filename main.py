import gym
import numpy as np
from utils import plot_learning_curve
from Actor import Agent

def main():
    # Creación del entorno
    env = gym.make('Pendulum-v1')
    n_actions = env.action_space.shape[0]

    # Creación del agente con el entorno y el número de acciones adecuados
    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=n_actions)
    n_games = 20  # Número de episodios a jugar

    # Archivo para guardar la gráfica de rendimiento
    figure_file = 'plots/pendulum.png'

    best_score = env.reward_range[0]  # Mejor puntuación inicializada con la peor posible
    score_history = []  # Lista para almacenar la puntuación en cada episodio
    load_checkpoint = False  # Bandera para cargar un punto de control previo

    # Si se carga un punto de control, se inicializan las transiciones en el búfer de repetición
    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            n_steps += 1
        agent.learn()
        agent.load_models()
        evaluate = True
    else:
        evaluate = False

    # Ciclo principal
    for i in range(n_games):
        observation = env.reset()[0]  # Reiniciar el entorno para un nuevo episodio
        done = False
        score = 0
        j = 0
        while not done and j < 500:
            action = agent.choose_action(observation, evaluate)  # Elegir una acción
            observation_, reward, done, info, _ = env.step(action)  # Realizar la acción en el entorno
            score += reward  # Actualizar la puntuación acumulada
            agent.remember(observation, action, reward, observation_[0], done)  # Almacenar la transición
            if not load_checkpoint:
                agent.learn()  # Aprender de la transición
            observation = observation_  # Actualizar el estado actual
            j += 1

        score_history.append(score)  # Almacenar la puntuación del episodio
        avg_score = np.mean(score_history[-100:])  # Calcular la puntuación media en los últimos 100 episodios

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

if __name__ == "__main__":
    main()
