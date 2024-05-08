import os
import tensorflow as tf
import keras.api as keras

class ActorNetwork(keras.Model):
        def __init__(self, layers_size = [50, 30], n_actions = 2, name='actor', chkpt_dir='tmp\ddpg'):
             #Inicializo la red, sus dimensiones y posibles acciones
             super(ActorNetwork, self).__init__()
             self.layer_1_dims = layers_size[0]
             self.layer_2_dims = layers_size[1]

             self.model_name = name
             self.checkpoint_dir = chkpt_dir
             #Guardamos el checkpoint del modelo en la dirección propia
             self.checkpoint_file = os.getcwd() + "\\model_weights\\" + self.model_name+ "_ddpg.weights.h5"

             self.layer_1 = keras.layers.Dense(self.layer_1_dims, activation='relu')
             self.layer_2 = keras.layers.Dense(self.layer_2_dims, activation='relu')
             self.mu = keras.layers.Dense(n_actions, activation='tanh')

        def call(self, state):
             layer_result = self.layer_1(state)
             layer_result = self.layer_2(layer_result)

             #Se puede multiplicar si el objetivo no es del rango de 1 a -1
             final_layer_result = self.mu(layer_result)

             return final_layer_result