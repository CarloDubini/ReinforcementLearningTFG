class ActorNetwork(keras.Model):
        def __init__(self, fc1_dims=512, fc2_dims=512, n_actions = 2, name='actor', chkpt_dir='tmp/ddpg'):
             #Inicializo la red, sus dimensiones y posibles acciones
             super(ActorNetwork, self).__init__()
             self.fc1_dims = fc1_dims
             self.fc2_dims = fc2_dims
             self.n_actions = n_actions

             self.model_name = name
             self.checkpoint_dir = chkpt_dir
             #Guardamos el checkpoint del modelo en la dirección propia
             self.checkpoint_file = os.path.join(self.checkpoint_dir,self.model_name+'_ddpg.h5')

             self.fc1 = Dense(self.fc1_dims, activation='relu')
             self.fc2 = Dense(self.fc2_dims, activation='relu')
             self.mu = Dense(self.n_actions, activation='tanh')

        def call(self, state):
             prob = self.fc1(state)
             prob = self.fc2(prob)

             #Se puede multiplicar si el objetivo no es del rango de 1 a -1
             mu = self.mu(prob)


             return mu