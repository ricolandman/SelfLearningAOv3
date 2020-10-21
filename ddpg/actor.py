import tensorflow as tf
import time
import time
import numpy as np
from keras.models import Sequential
from keras import backend as K
from keras.layers import Dense, Concatenate, BatchNormalization, Lambda, Flatten,LSTM,ConvLSTM2D
from keras.activations import tanh
from keras.models import Model,load_model
from keras.layers import Input,Conv2D,LocallyConnected2D,UpSampling2D,TimeDistributed,ZeroPadding2D
from keras.initializers import RandomUniform
from keras.regularizers import l2

class Actor(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate=0, tau=0, batch_size=1,
            clip_value=1,opt_length=1,init_length=0,action_scaling=1):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.clip_value = clip_value
        self.opt_length=opt_length
        self.action_scaling = action_scaling

        #Initialize actor network
        self.observations,self.prev_actions, self.out,self.actor_model = self.create_actor_network(self.action_scaling)
        self.actor_model.summary()

        self.network_params = tf.trainable_variables()
        print(self.network_params)

        #Initialize target network
        self.target_observations,self.target_prev_actions, self.target_out,\
                self.target_model = self.create_actor_network(self.action_scaling)

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]
        
        #Initialize stateful network (real-time controller)
        self.stateful_observations,self.stateful_prev_actions, self.stateful_out,\
                self.stateful_model = self.create_actor_network(self.action_scaling,stateful=True)
        self.stateful_network_params = tf.trainable_variables()[
            len(self.network_params)+len(self.target_network_params):]

        # Initialize action gradient (provided by the critic)
        self.action_gradient = tf.placeholder(tf.float32, [None]+ self.a_dim)
        
        #self.reg_action_gradients = [tf.add(grads,reg_factor*action) for grads,action in \
        #        zip(self.clipped_actor_gradients,self.network_params)]


        #Get gradients of actor network
        self.unnormalized_actor_gradients = tf.gradients(
                self.out[:,-opt_length:], self.network_params,
                -self.action_gradient[:,-opt_length:])

        #Divide gradients by batch size
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size*opt_length),
            self.unnormalized_actor_gradients))

        #Clip gradients
        self.clipped_actor_gradients = [tf.clip_by_value(grad, -self.clip_value,
            self.clip_value) for grad in self.actor_gradients]

        #This does not work yet!!! --> I think it does now
        reg_factor = 1e-2
        #self.clipped_actor_gradients[-1] = tf.add(self.clipped_actor_gradients[-1],-reg_factor*self.network_params[-1])
        #self.clipped_actor_gradients[-1] = reg_factor*self.network_params[-1]
        self.reg_actor_gradients = [tf.add(grads,reg_factor*params) if not 'bias' in \
                params.name else grads for grads,params in \
                zip(self.clipped_actor_gradients,self.network_params)]

        #Apply regularization
        #reg_factor = 
        #self.clipped_actor_gradients[-1] -= 

        #Define optimization op
        #self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
        #    apply_gradients(zip(self.reg_actor_gradients,self.network_params))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.clipped_actor_gradients,self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self,action_scaling,stateful=False):
        #Creates the actor model
        #The architecture has to be changed here!
        if stateful:
            observations = Input(batch_shape=[1,1]+self.s_dim[1:])
            prev_actions = Input(batch_shape = [1,1]+self.a_dim[1:])
        else:
            observations = Input(self.s_dim)
            prev_actions = Input(self.a_dim)
        x = Concatenate()([observations,prev_actions])
        #x = BatchNormalization()(x)
        x = ConvLSTM2D(8,(1,1),strides=1,padding='same',unit_forget_bias=True,stateful=stateful,\
                return_sequences=True)(x)
        #x = TimeDistributed(Conv2D(8,(3,3),strides=1,padding='same',activation='tanh'))(x)
        if not stateful:
            x = Lambda(lambda y: y[:,-self.opt_length:,:,:])(x)

        #x = TimeDistributed(Conv2D(8,(1,1),strides=1,padding='same',activation='relu'))(x)
        x = TimeDistributed(Conv2D(self.a_dim[-1],(1,1),strides=1,padding='same',\
                   kernel_initializer=RandomUniform(-3e-3,3e-3),activation='linear',\
                   use_bias=False))(x)
        #x = TimeDistributed(LocallyConnected2D(self.a_dim[-1],(1,1),strides=1,\
        #        activation='tanh',use_bias=False))(x)
        action = Lambda(lambda y: y*action_scaling)(x)
        #action = x
        model = Model(inputs=[observations,prev_actions],outputs=action)
        return(observations,prev_actions,action,model)

    def add_parameter_noise(self,noise):
        tensor_names = [tensor.name for tensor in self.actor_model.weights]
        #print(tensor_names)
        #time.sleep(1)
        weights_list = self.actor_model.get_weights()
        #perturbed_weights = [(1+noise*np.random.randn(*weights.shape))*weights if not \
        perturbed_weights = [(1+noise*np.random.randn(*weights.shape))*weights if not \
        #perturbed_weights = [weights+noise*np.random.randn(*weights.shape) if not \
                #('batch_normalization' in name)*('bias' in name)*('recurrent' in name)
                ('batch_normalization' in name)*('bias' in name)
                             else weights 
                             for (name,weights) in zip(tensor_names,weights_list)]
        self.stateful_model.set_weights(perturbed_weights)

    def train(self, inputs, a_gradient):
        #Performs one update of the actor
        self.sess.run(self.optimize, feed_dict={
            self.observations:inputs[0],
            self.prev_actions:inputs[1],
            self.action_gradient: a_gradient
        })

    def pretrain(self,gain):
        print('Pretraining actor')
        trials = 2000
        observations = np.random.randn(trials,20,*self.s_dim[1:])
        prev_actions = np.random.randn(trials,20,*self.a_dim[1:])
        Y = gain*observations[:,-self.opt_length:,:,:,0,np.newaxis]
        self.actor_model.compile(optimizer='adam',loss='mse')
        self.actor_model.fit([observations,prev_actions],Y,batch_size=4,epochs=1)
        self.hard_update_target_network()
        self.update_stateful_network()

    def predict(self, inputs):
        #Returns the action for the given state
        return self.sess.run(self.out, feed_dict={
            self.observations:inputs[0],
            self.prev_actions:inputs[1]
        })
    
    def predict_stateful(self, inputs):
        #Returns the action of the stateful model for the given state
        #return self.sess.run(self.stateful_out, feed_dict={
        #    self.stateful_observations:inputs[0],
        #    self.stateful_prev_actions:inputs[1]
        #})
        a = self.stateful_model.predict(inputs)
        return a[0,-1]

    def predict_target(self, inputs):
        #Returns the action of the target model for the given state
        return self.sess.run(self.target_out, feed_dict={
            self.target_observations: inputs[0],
            self.target_prev_actions: inputs[1]
        })

    def update_target_network(self):
        #Performs a soft target network update
        target_weights = self.target_model.get_weights()
        weights = self.actor_model.get_weights()
        self.target_model.set_weights([(1-self.tau)*target_weights[i]+self.tau*weights[i] for i in range(len(weights))])

    def hard_update_target_network(self):
        #Performs a hard target network update
        weights = self.actor_model.get_weights()
        self.target_model.set_weights(weights)
    
    def update_stateful_network(self):
        #Performs a hard stateful network update
        weights = self.actor_model.get_weights()
        self.stateful_model.set_weights(weights)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def load_model(self,model_name):
        #Loads weights from a trained model
        #Note: manually fix the architecture!
        loaded_model = load_model(model_name)
        weights = loaded_model.get_weights()
        self.actor_model.set_weights(weights)
        self.stateful_model.set_weights(weights)
        self.target_model.set_weights(weights)


