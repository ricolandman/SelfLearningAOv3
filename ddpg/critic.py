import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras import backend as K
from keras.layers import Dense, Concatenate, BatchNormalization, Lambda, Flatten,TimeDistributed
from keras.layers import Conv2D,LocallyConnected2D,UpSampling2D, LSTM, ConvLSTM2D,ZeroPadding2D
from keras.optimizers import Adam

from keras.activations import tanh
from keras.models import Model, load_model
from keras.layers import Input
from keras.initializers import RandomUniform
from keras.losses import mean_squared_error
from keras.regularizers import l2


class Critic(object):
    def  __init__(self, sess, state_dim, action_dim, q_dim, learning_rate, tau, gamma, num_actor_vars,
            init_length=1):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.q_dim = q_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.init_length=init_length

        # Create the critic network
        self.observations,self.prev_actions, self.action, self.out,self.model\
                = self.create_critic_network()
        self.model.compile(optimizer=Adam(self.learning_rate),loss='mse')
        self.model.summary()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_observations,self.target_prev_actions, self.target_action,\
                self.target_out,self.target_model = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]


        self.predicted_q_value = tf.placeholder(tf.float32, [None]+ self.q_dim)

        # Define loss and optimization Op
        l2 = 0
        #reg_loss = tf.nn.l2_loss(self.model.layers[-1].get_weights())
        reg_loss = tf.nn.l2_loss(self.network_params[-2])
        self.loss = mean_squared_error(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the critic with respect to the action
        self.action_grads = tf.gradients(self.out, self.action)
        #self.action_grads = tf.gradients(self.out[:,init_length:], self.action[:,init_length:])

    
    def create_critic_network(self):
        #Creates the actor model
        #The architecture has to be changed here!
        observations = Input(shape=self.s_dim)
        prev_actions = Input(shape=self.a_dim)
        action = Input(shape=self.a_dim)
        
        x = Concatenate()([observations,prev_actions])
        x = ConvLSTM2D(8,(3,3),strides=1,padding='same',unit_forget_bias=True,return_sequences=True)(x)
        x = Lambda(lambda y: y[:,self.init_length:,:,:])(x)
        #        output_shape=self.a_dim)(x)
        x = Concatenate()([x,action])
        x = TimeDistributed(Conv2D(8,(3,3),strides=1,padding='same',activation='relu'))(x)
        x = TimeDistributed(Conv2D(8,(3,3),strides=2,padding='same',activation='relu'))(x)
        #x = Conv2D(1,(3,3),strides=1,padding='same',activation='relu')(x)
        x = TimeDistributed(Flatten())(x)
        #x = TimeDistributed(Dense(32,activation='relu'))(x)
        value = TimeDistributed(Dense(1,kernel_initializer=RandomUniform(-3e-3,3e-3),kernel_regularizer=l2(1e-4)))(x)
        model = Model(inputs=[observations,prev_actions,action],outputs=value)
        return(observations,prev_actions,action,value,model)

    def train(self, inputs, action, predicted_q_value):
        #Performs one critic update
        #q_pred = self.model.predict_on_batch([inputs[0],inputs[1],action])
        loss = self.model.train_on_batch([inputs[0],inputs[1],action],predicted_q_value)
        return(loss)
        #return self.sess.run([self.out, self.optimize], feed_dict={
        #    self.observations: inputs[0],
        #    self.prev_actions: inputs[1],
        #    self.action: action,
        #    self.predicted_q_value: predicted_q_value
        #})

    def predict(self, inputs, action):
        #Predicts the discounted future reward
        return self.sess.run(self.out, feed_dict={
            self.observations: inputs[0],
            self.prev_actions: inputs[1],
            self.action: action
        })

    def predict_target(self, inputs, action):
        #Predicts the discounted future reward with the target network
        return self.sess.run(self.target_out, feed_dict={
            self.target_observations: inputs[0],
            self.target_prev_actions: inputs[1],
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        #Obtains the gradient with respect to the action for given state
        return self.sess.run(self.action_grads, feed_dict={
            self.observations: inputs[0],
            self.prev_actions: inputs[1],
            self.action: actions
        })

    def update_target_network(self):
        #Soft update of the target network
        target_weights = self.target_model.get_weights()
        weights = self.model.get_weights()
        self.target_model.set_weights([(1-self.tau)*target_weights[i]+self.tau*weights[i] for i in range(len(weights))])

    def hard_update_target_network(self):
        #Hard update of the target network
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def load_model(self,model_name):
        #Loads weights for previously trained critic
        #Note: manually fix the architecture!
        loaded_model = load_model(model_name)
        weights = loaded_model.get_weights()
        self.model.set_weights(weights)
        self.target_model.set_weights(weights)
