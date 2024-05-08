import os
import tensorflow as tf
import keras.api as keras

class ActorNetwork(keras.Model):
        def __init__(self, layers_size = [350, 350, 50], n_actions = 2, name='actor', chkpt_dir='tmp\ddpg'):
             #Inicializo la red, sus dimensiones y posibles acciones
             super(ActorNetwork, self).__init__()
             self.layer_1_dims = layers_size[0]
             self.layer_2_dims = layers_size[1]
             self.layer_3_dims = layers_size[2]
             self.n_actions = n_actions-1 #esto es para eliminar el griper de la red neuronal

             self.model_name = name
             self.checkpoint_dir = chkpt_dir
             #Guardamos el checkpoint del modelo en la dirección propia
             self.checkpoint_file = os.getcwd() + "\\model_weights\\" + self.model_name+ "_ddpg.weights.h5"

             self.layer_1 = keras.layers.Dense(self.layer_1_dims, activation='relu')
             self.layer_2 = keras.layers.Dense(self.layer_2_dims, activation='relu')
             self.layer_3 = keras.layers.Dense(self.layer_3_dims, activation='relu')
             self.mu = keras.layers.Dense(self.n_actions, activation='tanh')

        def call(self, state):
             layer_result = self.layer_1(state)
             layer_result = self.layer_2(layer_result)
             layer_result = self.layer_3(layer_result)

             #Se puede multiplicar si el objetivo no es del rango de 1 a -1
             final_layer_result = self.mu(layer_result)
             #esto se debe a que aunque no se necesite, el entorno te exige entregar el resultado del griper, cosa que es malo para el funcionamiento de la NN.

             add_gripper_act = tf.shape(final_layer_result)[0]
             add_gripper_act = tf.fill((add_gripper_act, 1), 0.1)
             final_layer_result = tf.concat([final_layer_result, add_gripper_act], axis=1)

             return final_layer_result