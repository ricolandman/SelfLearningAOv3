import tensorflow as tf
import os
import time
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from ddpg.replay_buffer import *
from ddpg.actor import *
from ddpg.critic import *
from ddpg.ou_noise import *
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
#import asdf
import concurrent.futures

class DataCollector():
    def __init__(self,env,replay_buffer,action_dim):
        self.h_o = deque()
        self.h_a = deque()
        self.h_r = deque()
        self.strehls = deque()
        self.contrasts= deque()
        self.env = env
        self.replay_buffer = replay_buffer
        self.action_dim = action_dim

    def run(self,sess,actor,noise,num_iterations=500):
        with sess.graph.as_default():
            start_time = time.time()
            
            #Decay noise and learning rates
            ep_reward = 0
            ep_ave_max_q = 0
            
            #Clear buffers
            self.h_o.clear()
            self.h_a.clear()
            self.h_r.clear()
            self.strehls.clear()
            self.contrasts.clear()

            #Reset environment and get initial observation 
            o = self.env.reset(reset_dm=True,reset_turbulence=True)
            self.h_o.append(o)
            a = np.zeros(self.action_dim[1:])
            self.iteration=0

            #Reset stateful actor model
            actor.stateful_model.reset_states()

            for self.iteration in range(1,num_iterations+1):
                a = actor.predict_stateful([o[np.newaxis,np.newaxis,:],a[np.newaxis,np.newaxis,:]])
                #Add exploration noise
                a = a + noise*np.random.randn(*a.shape)

                #Clip action
                a = np.clip(a,-1,1)
                a = 0.5*o
                o,r,terminate = self.env.step(a)

                self.h_o.append(o)
                self.h_a.append(a)
                self.h_r.append(r)

                ep_reward += np.sum(r)
                self.strehls.append(self.env.strehl)
                self.contrasts.append(self.env.contrast)
            
            self.replay_buffer.add(np.array(self.h_o),np.array(self.h_a),np.array(self.h_r))
            average_strehl = np.mean(np.array(self.strehls)[50:])
            average_contrast = np.mean(np.array(self.contrasts)[50:])
            #self.average_strehls.append(np.mean(np.array(strehls)))
            #self.total_reward.append(np.sum(ep_reward))
            end_time = time.time()
            print('| Average Strehl: {0:.3f} | Total Contrast: {1:.2e} | Total reward: {2:.1f} | Time: {3:.2f} s '\
                    .format(average_strehl,average_contrast,np.sum(ep_reward),end_time-start_time))
        return self.replay_buffer,average_strehl,average_contrast

class Trainer():
    def __init__(self,actor,critic,batch_size,opt_length,init_length,actuator_mask):
        self.actor = actor
        self.critic = critic
        self.batch_size = batch_size
        self.length = opt_length+init_length
        self.opt_length = opt_length
        self.actuator_mask = actuator_mask.reshape(25,25,1)
        self.init_length = init_length
        self.mse = deque()

    def train(self,sess,replay_buffer,n_iter,noise):
        #print('Start training')
        self.noise = noise
        start_time = time.time()
        self.mse.clear()
        with sess.graph.as_default():
            start_time = time.time()
            for i in range(n_iter):
                self.sample_batch(replay_buffer)
                self.train_critic()
                self.train_actor()
        end_time = time.time()
        print(f'Trained on {n_iter} batches in {end_time-start_time:.1f} seconds | Critic MSE: {np.mean(np.array(self.mse)):.5f}')
        return self.actor
    
    def sample_batch(self,replay_buffer):
        #Samples a batch (s,a,r,s') from the replay buffer
        self.s_batch, self.a_batch, self.r_batch, self.s2_batch = replay_buffer.sample_batch(
            self.batch_size,self.length)

    def train_critic(self):
        #Trains critic
        target_a = self.actor.predict_target(self.s2_batch)
        #target_a = self.actor.predict_target(self.s2_batch)*self.actuator_mask
        target_q = self.critic.predict_target(self.s2_batch,target_a+self.noise*np.random.randn(*target_a.shape))
        self.r_batch = self.r_batch[:,-self.opt_length:,np.newaxis]

        #Calculate critic targets
        y = self.r_batch + self.critic.gamma*target_q

        #Perform update step
        loss = self.critic.train(self.s_batch, self.a_batch[:,-self.opt_length:], y)

        #Update target network
        self.critic.update_target_network()

        #Save statistics
        #if self.make_debug_plots:
        #    self.y.append(y)
        #    self.q_pred.append(predicted_q_value)
        #    self.ave_q = np.mean(np.max(predicted_q_value,axis=-1))
        self.mse.append(loss)

    def train_actor(self):
        #Trains the actor
        a_outs = self.actor.predict(self.s_batch)
        #a_outs = self.action_scaling*a_outs*np.expand_dims(self.env.actuator_mask,axis=2)
        #a_outs = a_outs*np.expand_dims(self.env.actuator_mask,axis=2)
        #Get critic gradient
        grads = self.critic.action_gradients(self.s_batch, a_outs)
        
        #if self.make_debug_plots:
        #    self.a_outs.append(a_outs)
        #    self.grads.append(grads[0])

        #Update actor
        #self.actor.train(self.s_batch, grads[0]*self.env.actuator_mask[:,:,np.newaxis])
        #self.actor.train(self.s_batch, grads[0]*self.actuator_mask)
        self.actor.train(self.s_batch, grads[0])
        self.actor.update_target_network()
        self.actor.update_stateful_network()


class DDPG_agent():
    def __init__(self,env,args):

        #Set agent parameters (See main.py for description)
        self.args = args
        self.min_episodes = args['min_episodes']
        self.max_episodes = args['max_episodes']
        self.num_iterations = args['iterations_per_episode']
        self.noise =args['start_noise']
        self.minibatch_size = args['minibatch_size']
        self.opt_length= args['optimization_length']
        self.init_length= args['initialization_length']
        self.early_stopping = args['early_stopping']
        self.patience = args['patience']
        self.trajectory_length = args['trajectory_length']
        self.warmup = args['warmup']
        self.critic_warmup = args['critic_warmup']
        self.noise_decay = args['noise_decay']
        self.savename = args['savename']
        self.save_interval = args['save_interval']
        self.start_actor_lr = args['actor_lr']
        self.start_critic_lr = args['critic_lr']
        self.actor_lr_decay = args['actor_lr_decay']
        self.critic_lr_decay = args['critic_lr_decay']
        self.make_debug_plots = args['make_debug_plots']
        self.noise_type = args['noise_type']
        self.action_scaling = args['action_scaling']
        self.use_integrator = args['use_integrator']
        self.use_stateful_actor = args['use_stateful_actor']
        self.pretrain_actor = args['pretrain_actor']
        self.pretrain_gain = args['pretrain_gain']
        self.num_training_batches = args['num_training_batches']
        self.env = env

        #Initialize tensorflow session
        self.sess=tf.Session()
        K.set_session(self.sess)

        #Set action,state and reward dimensions
        if self.env.control_mode=='modal':
            self.action_dim = [None,self.env.num_act,self.env.num_act,1]
            self.state_dim = self.action_dim
        elif self.env.control_mode=='state':
            self.action_dim = [None,self.env.N_mla,self.env.N_mla,2]
            self.state_dim = self.action_dim
        elif self.env.control_mode=='both':
            self.action_dim = [None,self.env.N_mla,self.env.N_mla,2]
            self.state_dim = [None,self.env.num_act,self.env.num_act,1]
        q_dim = [None,1]

        print('Initializing actor')
        self.actor = Actor(self.sess, self.state_dim, self.action_dim,
                            float(args['actor_lr']), float(args['tau']),
                            int(args['minibatch_size']),args['actor_grad_clip'],
                            args['optimization_length'],args['initialization_length'])

        print('Initializing critic')
        self.critic = Critic(self.sess, self.state_dim, self.action_dim, q_dim,
                            float(args['critic_lr']), float(args['tau']),
                            float(args['gamma']),
                            self.actor.get_num_trainable_vars(),self.init_length)

        #Initialize replay buffer
        self.replay_buffer = ReplayBuffer(int(args['buffer_size']),self.num_iterations)

        #Initalize network parameters
        self.sess.run(tf.global_variables_initializer())
        self.actor.hard_update_target_network()
        self.critic.hard_update_target_network()

        #Load models
        if args['load_model']:
            self.actor.load_model('./models/'+args['load_model_name']+'_actor.hdf5')
            self.critic.load_model('./models/'+args['load_model_name']+'_critic.hdf5')
            print('Loaded weights from', args['load_model_name'])

    def train(self,run_async=True):
        K.set_learning_phase(0)
        
        #Save hyperparameters to file
        if self.pretrain_actor:
            self.actor.pretrain(self.pretrain_gain)
        
        data_collector = DataCollector(self.env,self.replay_buffer,self.action_dim)
        trainer = Trainer(self.actor,self.critic,self.minibatch_size,self.opt_length,self.init_length,self.env.actuator_mask)

        strehls = []
        contrasts = []
        if run_async:
            for episode in range(self.max_episodes):
                print('\n Episode:',episode)
                time.sleep(0.01)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    f1 = executor.submit(data_collector.run, self.sess,self.actor,self.noise,self.num_iterations)
                    if episode>self.warmup:
                        f2 = executor.submit(trainer.train, self.sess,self.replay_buffer,self.num_training_batches,self.noise)
                
                    self.replay_buffer,strehl,contrast = f1.result()
                    strehls.append(strehl)
                    contrasts.append(contrast)
                    if episode>self.warmup:
                        self.actor = f2.result()
                self.noise *= self.noise_decay

                if episode%self.save_interval==0:
                    self.save(self.savename)
                    train_curve = pd.DataFrame()
                    train_curve['Episode'] = np.arange(1,len(strehls)+1)
                    train_curve['Strehl'] = np.array(strehls)
                    train_curve['Contrast'] = np.array(contrasts)
                    train_curve.to_csv('results/logs/{0}_traincurve.csv'.format(self.savename)\
                            ,sep=',')
        else:
            for episode in range(self.max_episodes):
                print('\n Episode:',episode)
                self.replay_buffer = data_collector.run(self.sess,self.actor,self.noise,self.num_iterations)
                self.actor = trainer.train(self.sess,self.replay_buffer,self.num_training_batches,self.noise)
                self.noise *= self.noise_decay
                if episode%self.save_interval==0:
                    self.save(self.savename)


    def save(self,savename):
        #Save actor and critic for testing or further training
        savepath = './models/'+savename
        #self.actor.actor_model.save(savepath+"_actor.hdf5")
        self.actor.stateful_model.save(savepath+"_actor.hdf5")
        #self.critic.model.save(savepath+"_critic.hdf5")
        print("Saved actor model: %s" % savepath)
        #print("Saved critic model: %s" % savepath)
