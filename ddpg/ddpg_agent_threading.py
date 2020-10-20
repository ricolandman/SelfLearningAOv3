import tensorflow as tf
import os
import time
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from ddpg.replay_buffer import *
from ddpg.actor import *
from ddpg.critic import *
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
#import asdf
import concurrent.futures

class DataCollector():
    def __init__(self,env,replay_buffer,action_dim,noise_type,use_integrator=False,integrator_gain=None,\
                use_stateful_actor=False, non_stateful_state_length=1):
        self.h_o = deque()
        self.h_a = deque()
        self.h_r = deque()
        self.strehls = deque()
        self.contrasts= deque()
        self.env = env
        self.replay_buffer = replay_buffer
        self.action_dim = action_dim
        self.use_integrator = use_integrator
        self.integrator_gain = integrator_gain
        self.noise_type = noise_type
        self.use_stateful_actor = use_stateful_actor
        self.non_stateful_state_length = non_stateful_state_length

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
            
            if not self.use_stateful_actor:
                for _ in range(self.non_stateful_state_length+1):
                    self.h_o.append(o)
                    self.h_a.append(a)

            for self.iteration in range(1,num_iterations+1):
                if self.use_integrator:
                    a = self.integrator_gain*o[:,:,0,np.newaxis] 
                elif self.use_stateful_actor:
                    if noise>0 and self.noise_type=='parameter' and self.iteration%50==0:
                        actor.add_parameter_noise(noise)
                    a = actor.predict_stateful([o[np.newaxis,np.newaxis,:],a[np.newaxis,np.newaxis,:]])
                else:
                    #Check this!! --> I think this is now OK
                    o_non_stateful = np.array([self.h_o[x] for x in np.arange(-self.non_stateful_state_length-1,0)])
                    a_non_stateful = np.array([self.h_a[x] for x in np.arange(-self.non_stateful_state_length-1,0)])
                    a = actor.predict([o_non_stateful[np.newaxis,:],a_non_stateful[np.newaxis,:]])[0,-1]
                
                if self.noise_type=='action':
                    #Add exploration noise
                    #for _ in range(10):
                    #    noisy_act = np.unravel_index(np.random.choice(a.size),a.shape)
                    #    a[noisy_act] += np.random.randn()
                    a = a + np.random.uniform(low=0,high=noise)*np.random.randn(*a.shape)

                #Clip action
                #a = np.clip(a,-self.action_scaling,self.action_scaling)
                o,r,terminate = self.env.step(a)

                self.h_o.append(o)
                self.h_a.append(a)
                self.h_r.append(r)

                ep_reward += np.sum(r)
                self.strehls.append(self.env.strehl)
                self.contrasts.append(self.env.contrast)
            
            self.replay_buffer.add(np.array(self.h_o),np.array(self.h_a),np.array(self.h_r))
            average_strehl = np.mean(np.array(self.strehls)[50:])
            #print(average_strehl)
            #average_strehl = self.env.science_image.max()
            average_contrast = np.mean(np.array(self.contrasts)[50:])
            #self.average_strehls.append(np.mean(np.array(strehls)))
            #self.total_reward.append(np.sum(ep_reward))
            end_time = time.time()
            print('| Average Strehl: {0:.3f} | Total Contrast: {1:.2e} | Total reward: {2:.1f} | Time: {3:.2f} s '\
                    .format(average_strehl,average_contrast,np.sum(ep_reward),end_time-start_time))
        return self.replay_buffer,average_strehl,average_contrast

class Trainer():
    def __init__(self,actor,critic,batch_size,opt_length,init_length):
        self.actor = actor
        self.critic = critic
        self.batch_size = batch_size
        self.length = opt_length+init_length
        self.opt_length = opt_length
        self.init_length = init_length
        self.mse = deque()

    def train(self,sess,replay_buffer,n_iter,noise,train_actor=True):
        #print('Start training')
        self.noise = noise
        start_time = time.time()
        self.mse.clear()
        with sess.graph.as_default():
            start_time = time.time()
            for i in range(n_iter):
                self.sample_batch(replay_buffer)
                self.train_critic()
                if train_actor:
                    self.train_actor()
            self.actor.update_stateful_network()
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
        target_q = self.critic.predict_target(self.s2_batch,target_a)
        #target_q = self.critic.predict_target(self.s2_batch,target_a+self.noise*np.random.randn(*target_a.shape))
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
        #Get critic gradient
        grads = self.critic.action_gradients(self.s_batch, a_outs)
        
        #if self.make_debug_plots:
        #    self.a_outs.append(a_outs)
        #    self.grads.append(grads[0])

        #Update actor
        self.actor.train(self.s_batch, grads[0])
        self.actor.update_target_network()
        #self.actor.update_stateful_network()


class DDPG_agent():
    def __init__(self,env,args):

        #Set agent parameters (See main.py for description)
        self.max_episodes = args['max_episodes']
        self.num_iterations = args['iterations_per_episode']
        self.noise = args['start_noise']
        self.minibatch_size = args['minibatch_size']
        self.opt_length= args['optimization_length']
        self.init_length= args['initialization_length']
        self.early_stopping = args['early_stopping']
        self.patience = args['patience']
        self.warmup = args['warmup']
        self.actor_warmup = args['actor_warmup']
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
        self.integrator_gain = args['integrator_gain']
        self.use_stateful_actor = args['use_stateful_actor']
        self.pretrain_actor = args['pretrain_actor']
        self.pretrain_gain = args['pretrain_gain']
        self.num_training_batches = args['num_training_batches']
        self.reward_type = args['reward_type']
        self.env = env

        #Initialize tensorflow session
        self.sess=tf.Session()
        K.set_session(self.sess)

        #Set action,state and reward dimensions
        if self.env.control_mode=='modal':
            self.action_dim = [None,self.env.num_act,self.env.num_act,1]
            self.state_dim = [None,self.env.num_act,self.env.num_act,1]
            #self.state_dim = self.action_dim
        elif self.env.control_mode=='state':
            self.action_dim = [None,self.env.N_mla,self.env.N_mla,2]
            self.state_dim = self.action_dim
        elif self.env.control_mode=='both':
            self.action_dim = [None,self.env.N_mla,self.env.N_mla,2]
            self.state_dim = [None,self.env.num_act,self.env.num_act,1]
        if self.reward_type=='modal':
            q_dim = [None,self.env.num_act,self.env.num_act,1]
        else:
            q_dim = [None,1]

        print('Initializing actor')
        self.actor = Actor(self.sess, self.state_dim, self.action_dim,
                            float(args['actor_lr']), float(args['tau']),
                            int(args['minibatch_size']),args['actor_grad_clip'],
                            args['optimization_length'],args['initialization_length'],
                            args['action_scaling'])

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
        if self.pretrain_actor and not self.use_integrator:
            self.actor.pretrain(self.pretrain_gain)
        
        data_collector = DataCollector(self.env,self.replay_buffer,self.action_dim,self.noise_type,self.use_integrator,
        self.integrator_gain, self.use_stateful_actor,self.opt_length+self.init_length)
        trainer = Trainer(self.actor,self.critic,self.minibatch_size,self.opt_length,self.init_length)

        self.strehls = []
        self.contrasts = []
        for episode in range(self.max_episodes):

            #Test policy without noise once in a while
            if episode%10==0 and episode>1:
                print('Evaluating current policy without noise...')
                self.replay_buffer,strehl,contrast = data_collector.run(self.sess,self.actor,0,self.num_iterations)
                if self.make_debug_plots:
                    self.debug_plots()

            #If run_async we collect data and train in the same time
            #This is especially useful when a GPU is available
            if run_async:

                print('\n Episode:',episode)
                time.sleep(0.01)
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    f1 = executor.submit(data_collector.run, self.sess,self.actor,self.noise,self.num_iterations)
                    if episode>self.warmup:
                        train_actor = episode>self.actor_warmup
                        f2 = executor.submit(trainer.train, self.sess,self.replay_buffer,\
                                self.num_training_batches,self.noise,train_actor)
                
                    self.replay_buffer,strehl,contrast = f1.result()
                    if episode>self.warmup:
                        self.actor = f2.result()
                

            else:
                print('\n Episode:',episode)
                #Collect data
                self.replay_buffer,strehl,contrast = data_collector.run(self.sess,self.actor,self.noise,self.num_iterations)
                #Train actor and critic
                if episode>self.warmup:
                    self.actor = trainer.train(self.sess,self.replay_buffer,self.num_training_batches,self.noise)
            #Decay noise
            self.strehls.append(strehl)
            self.contrasts.append(contrast)
            self.noise *= self.noise_decay

            #Save
            if episode%self.save_interval==0:
                self.save(self.savename)
                train_curve = pd.DataFrame()
                train_curve['Episode'] = np.arange(1,len(self.strehls)+1)
                train_curve['Strehl'] = np.array(self.strehls)
                train_curve['Contrast'] = np.array(self.contrasts)
                train_curve.to_csv('results/logs/{0}_traincurve.csv'.format(self.savename)\
                        ,sep=',')

            #Make debug plots
            if self.make_debug_plots:
                self.debug_plots()


    def save(self,savename):
        #Save actor and critic for testing or further training
        savepath = './models/'+savename
        #self.actor.actor_model.save(savepath+"_actor.hdf5")
        self.actor.stateful_model.save(savepath+"_actor.hdf5")
        #self.critic.model.save(savepath+"_critic.hdf5")
        print("Saved actor model: %s" % savepath)
        #print("Saved critic model: %s" % savepath)

    def debug_plots(self):
        print('Making debug plots')
        plt.figure(1,figsize=(12,8))
        plt.clf()
        plt.subplot(2,3,1)
        plt.title('Strehl')
        plt.plot(self.strehls,color='orangered',label='Strehl')
        plt.subplot(2,3,2)
        plt.plot(self.contrasts,color='darkblue',label='Contrast')
        plt.yscale('log')
        plt.title('Contrast')
        plt.subplot(2,3,3)
        plt.title('Focal plane image')
        plt.imshow(np.log10(self.env.science_coro_image.reshape(self.env.focal_pixels,\
            self.env.focal_pixels)/self.env.science_image.max()),vmin=-4,vmax=-1,cmap='afmhot')
        plt.colorbar()
        plt.subplot(2,3,4)
        plt.title('Wavefront variance')
        plt.imshow(np.sqrt(np.mean(self.env.phase_screens**2,axis=0)),cmap='Reds')
        plt.colorbar()
        plt.subplot(2,3,5)
        plt.title('Mean residual phase screen')
        plt.imshow(np.mean(self.env.phase_screens,axis=0),cmap='bwr')
        plt.colorbar()
        plt.subplot(2,3,6)
        plt.title('Mean DM shape')
        plt.imshow(np.mean(self.env.dm_shapes,axis=0),cmap='bwr')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('results/plots/debug_plots_{0}.png'.format(self.savename))
        plt.draw()
        #plt.close()
        plt.pause(0.0001)

