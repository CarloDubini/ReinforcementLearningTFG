import tensorflow as tf
from keras.api._v2.keras.optimizers import Adam
from keras.api._v2.keras.layers import Dense
import numpy as np

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork

class Agent:
    """El agente DDPG, que incluye una red crítica y una red de actor, generalizado para su uso en cualquier entorno sabiendo los parámetros de entrada (Sensores en REAS) y número de acciones a tomar"""
    def __init__(self, input_dims, alpha=0.001, beta=0.002, env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005, fc1=400, fc2=300, batch_size=64, noise=0.1):
        """Utiliza alpha y beta para los porcentajes de aprendizaje de las redes, y gamma para cambiar el objetivo de la red crítica, así como tau se utiliza para controlar la tasa a la que se actualizan los pesos de la red objetivo.
        Los fc's sirven para el número de neuronas de las redes. n_actions es la cantidad de acciones finales a elegir.
        Tamaño máximo es para el replay buffer y """
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.actor = ActorNetwork(n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(name='critic')
        self.target_actor = ActorNetwork(n_actions=n_actions, name='target_actor')
        self.target_critic = CriticNetwork(name='target_critic')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        """Actualiza los valores de peso según tau, controlando esta a qué velocidad se actualizan"""
        if tau is None:
            tau = self.tau
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        """Guarda la transición en la memoria de replay"""
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print('.... guardando modelos ....')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print('.... cargando modelos ....')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def choose_action(self, observation, evaluate=False):
        """Utiliza la red neuronal del actor para predecir acciones, opcionalmente añade ruido para la exploración y
        garantiza que las acciones generadas estén dentro de los límites permitidos por el entorno."""
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=self.noise)
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions[0]
