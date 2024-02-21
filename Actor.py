import tensorflow as tf
from keras.api._v2.keras.optimizers import Adam
from keras.api._v2.keras.layers import Dense
import numpy as np
import keras.api._v2.keras as keras

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
        fc_dims= float(round((input_dims[0]+n_actions)*0.6)) # numero de neuronas escondidas en cada capa
        self.actor = ActorNetwork(n_actions=n_actions, name='actor',fc1_dims=fc_dims,fc2_dims=fc_dims)
        self.critic = CriticNetwork(name='critic',fc1_dims=fc_dims,fc2_dims=fc_dims)
        self.target_actor = ActorNetwork(n_actions=n_actions, name='target_actor',fc1_dims=fc_dims,fc2_dims=fc_dims)
        self.target_critic = CriticNetwork(name='target_critic',fc1_dims=fc_dims,fc2_dims=fc_dims)

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
        observation = np.array(observation)  # Convertir observation a una matriz numpy
        state = tf.convert_to_tensor([observation], dtype=np.float32)
        actions = self.actor(state)
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=self.noise)
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions[0]
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)
        
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(states_, target_actions),1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = rewards + self.gamma*critic_value_*(1-done)
            critic_loss = keras.losses.MSE(target, critic_value)
        
            critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)

        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))
        
        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

            actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)

        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters()

        

        



