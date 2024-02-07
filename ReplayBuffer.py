import os
import random
import numpy as np
import tensorflow as tf
import keras.api._v2.keras as keras
from keras.api._v2.keras.optimizers import Adam
from keras.api._v2.keras.layers import Dense

class ReplayBuffer:
  """El búfer de repetición es una estructura que almacena un número fijo de experiencias o transiciones pasadas. Estas consisten
   en información sobre el estado, acción, recompensa, estado siguiente y si ha terminado. La principal ventaja de su uso es que
  permite al agente romper la correlación entre experiencias consecutivas, reduciendo así el impacto de las correlaciones
  temporales que perjudican al rendimiento."""
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        #Inicializo los vectores de memoria para el tipo de input y el tamaño fijo dado
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros((self.mem_size))
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, new_state, done):
      """Al guardar las transiciones es importante saber que solo caben mem_size, lo cual hace que se reescriban en FIFO"""
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward

        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
      """Devuelve una sample de las almacenadas en forma de tupla ( states, actions, rewards, states_, dones)"""
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones