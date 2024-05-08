
import os
import tensorflow as tf
import keras.api as keras

class CriticNetwork(keras.Model):
    def __init__(self, layers_size = [50, 30], name='critic', chkpt_dir='tmp\ddpg'):
        #Inicializo la red, sus dimensiones y posibles acciones
        super(CriticNetwork, self).__init__()
        self.layers_1_dims = layers_size[0]
        self.layers_2_dims = layers_size[1]

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        #Guardamos el checkpoint del modelo en la dirección propia
        self.checkpoint_file = os.getcwd() + "\\model_weights\\" + self.model_name+ "_ddpg.weights.h5"

        #Creo las tres redes que van a ser las capas de mi red
        self.layer_1 = keras.layers.Dense(self.layers_1_dims, activation='relu')
        self.layer_2 = keras.layers.Dense(self.layers_2_dims, activation='relu')
        self.final_layer = keras.layers.Dense(1, activation=None)

    def call(self, state, action):
        action_value = self.layer_1(tf.concat([state,action], axis=1))
        action_value = self.layer_2(action_value)

        final_val = self.final_layer(action_value)

        return final_val