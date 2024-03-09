import os
import tensorflow as tf
from keras.api._v2.keras.layers import Dense
import keras.api._v2.keras as keras

class ActorNetwork(keras.Model):
        def __init__(self, fc1_dims=512, fc2_dims=512, n_actions = 2, name='actor', chkpt_dir='tmp\ddpg'):
             #Inicializo la red, sus dimensiones y posibles acciones
             super(ActorNetwork, self).__init__()
             self.fc1_dims = fc1_dims
             self.fc2_dims = fc2_dims
             self.n_actions = n_actions-1 #esto es para eliminar el griper de la red neuronal

             self.model_name = name
             self.checkpoint_dir = chkpt_dir
             #Guardamos el checkpoint del modelo en la dirección propia
             self.checkpoint_file = os.getcwd() + "\\model_weights\\" + self.model_name+ "_ddpg.h5"

             self.fc1 = Dense(self.fc1_dims, activation='relu')
             self.fc2 = Dense(self.fc2_dims, activation='relu')
             self.mu = Dense(self.n_actions, activation='tanh')

        def call(self, state):
             prob = self.fc1(state)
             prob = self.fc2(prob)

             #Se puede multiplicar si el objetivo no es del rango de 1 a -1
             mu = self.mu(prob)
             #esto se debe a que aunque no se necesite, el entorno te exige entregar el resultado del griper, cosa que es malo para el funcionamiento de la NN.

             nc = tf.shape(mu)[0]
             nc = tf.fill((nc, 1), 0.1)
             mu = tf.concat([mu, nc], axis=1)

             return mu