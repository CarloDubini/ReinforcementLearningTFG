

import tensorflow as tf
from keras.api._v2.keras.layers import Dense
import keras.api._v2.keras as keras

class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, name='critic', chkpt_dir='tmp/ddpg'):
        #Inicializo la red, sus dimensiones y posibles acciones
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        #Guardamos el checkpoint del modelo en la dirección propia
        self.checkpoint_file = os.path.join(self.checkpoint_dir,self.model_name+'_ddpg.h5')

        #Creo las tres redes que van a ser las capas de mi red
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        action_value = self.fc1(tf.concat([state,action], axis=1))
        action_value = self.fc2(action_value)

        q = self.q(action_value)

        return q