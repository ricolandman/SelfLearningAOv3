import argparse
import time
import csv
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)
import pprint as pp
from ao.ao_env import *
from ddpg.ddpg_agent_threading import *

np.random.seed(12345)
#--------------Make AO environment-----------------

def reward_function(strehl,contrast,modal_res=None):
    #NOTE: Scaling the reward also scales the actor gradient so change the learning rate accordingly!
    #reward = np.log(strehl)
    reward = -(modal_res)**2
    return reward

env_params = dict()
env_params['D'] = 8  #Diameter of simulated telescope
env_params['wavelength_science'] = 1.65e-6   #Wavelength (m)
env_params['wavelength_sensing'] = 0.7e-6   #Wavelength (m) of monochromatic source
env_params['reward_function'] = reward_function    #Reward function
env_params['pupil_pixels'] = 240  #Number of pixels in pupil_plane
env_params['num_iterations'] = 1000  #Number of iterations per episode
env_params['burnin_iterations'] = 200  #Number of iterations before integrating science image
env_params['show_image'] = True #Visualize wavefronts for debugging of simulations, this is too slow for real training
env_params['verbosity'] = True      #Print progress
env_params['num_airy'] = 24  #Size of focal plane in number of airy rings
env_params['num_photons'] = np.inf #Number of photons in incoming beam for adding photon noise
#env_params['wfs_type'] = 'phase'
env_params['wfs_type'] = 'pyramid'
env_params['wfs_type'] = 'shack-hartmann'
env_params['closed_loop_freq'] = 1380   #Frequency (Hz) to run the correction
#env_params['closed_loop_freq'] = 1380   #Frequency (Hz) to run the correction
env_params['servo_lag'] = 2.2/env_params['closed_loop_freq']
env_params['temp_oversampling'] = 1
env_params['stellar_magnitude'] = 5

env_params['turbulence_mode'] = 'atmosphere'
#Turbulence mode, options:
#   dm: Turbulence only consists of combination of mirror modes
#   atmosphere: Atmospheric turbulence with fixed wind velocity and angle
#   atmosphere_random: Two-layered atmospheric turbulence with random wind velocity and angle for every episode

env_params['L0'] = [50,50]  #Outer scale parameter of simulated atmospheric layer
env_params['Cn2'] = [Cn_squared_from_fried_parameter(r0=0.15, wavelength=500e-9)]
#env_params['Cn2'] = [1e-13,1.3e-14]
env_params['angles'] = [0,45]
env_params['velocity'] = [15,30]  #Wind velocity vector
#env_params['velocity'] = [12,30]  #Wind velocity vector
#env_params['velocity'] = [0,0]  #Wind velocity vector
env_params['heights'] = [0,11e3]
env_params['scintillation'] = False
#env_params['heights'] = [0]

env_params['num_actuators'] = 41 #Number of actuators of the DM along pupil diameter
env_params['N_mla'] = 40 #Number of microlenses in Shack-Hartmann WFS

env_params['control_mode'] = 'modal'
#Control mode, options:
#   modal: Controller takes as input reconstructed modal coefficients and returns modal DM commands.
#   state: Controller takes as input slope measurements and returns slope measurements to correct for.
#   both: Controller takes as input slope measurements and returns modal DM commands.
#NOTE: If 'both' is used the architecture needs to be adjusted accordingly.

print('Initializing AO environment')
env = AO_env(env_params)

#--------------Hyperparameters of networks------------
#Architectures of networks can be changed in ddpg/actor.py and ddpg/critic.py.
params = dict()
params['actor_lr'] = 3e-6       #Learning rate for the actor
params['actor_lr_decay'] = np.exp(np.log(0.5)/50.)

params['critic_lr'] = 1e-3      #Learning rate for the critic
params['critic_lr_decay'] = np.exp(np.log(0.5)/30.)

params['tau'] = 1e-3      #Update parameter for the target networks: Q' = (1-tau)Q' + tau Q 
params['actor_grad_clip'] = 1.   #Clipping of the gradient for updating the actor to avoid large changes
params['use_stateful_actor'] = True
params['pretrain_actor'] = True
params['pretrain_gain'] = 0.4

#--------------RL algorithm hyperparameters-----------
params['gamma'] = 0.95 #Discount factor for expected future rewards
params['reward_type'] = 'modal'
params['buffer_size'] = 50 #Maximum number of episodes to save in the replay buffer
params['minibatch_size'] = 16 #Batch size for training critic and actor
params['num_training_batches'] = 1000  #Number of batches to train on every episode
params['optimization_length'] = 10 #Size of the history/number of steps to use in the BPTT
params['initialization_length'] = 10
params['trajectory_length'] = 1     #Number of steps to use observed rewards instead of bootstrapping with target critic.
#This does not work properly at ends op episodes and should be 1.

params['warmup'] = 0
params['actor_warmup'] = 0
params['action_scaling'] = 3     #Maximum possible action
params['use_integrator'] = True
params['integrator_gain'] = -0.5
#Probability of randomly using integrator for an iteration to fill the replay buffer with 'good' experience.
#This can be used to improve training stability. 
#Gain of the integrator is defaulted to 0.3 and can be set in ddpg/ddpg_agent.py

#-------------Closed Loop parameters------------------
params['iterations_per_episode'] = env_params['num_iterations']

#-------------Exploration noise parameters----------

params['start_noise'] = 0. #standard deviation of the random action noise
#params['noise_type'] = 'gaussian'       #Action noise type: gaussian or ou
params['theta_noise'] = 0.8             #Only used for ou noise
params['noise_decay'] = np.exp(np.log(0.5)/10)            #noise decay factor
params['noise_type'] = 'action'    #wether to use action or parameter space noise for exploration

#---------------Save/load parameters-------------------
params['max_episodes']= 1000            #Number of episodes to train for

params['savename'] = "newcond_large_meansubtract_morecriticlstm_8"

#The actor will be saved to ./models/{savename}_{episode}_actor.hdf5 
#The critic will be saved to ./models/{savename}_{episode}_critic.hdf5 
#The reward curve will be saved to ./logs/{savename}_train_curve.txt

params['save_interval'] = 10    #Number of episodes inbetween saves
params['early_stopping']=False  #Wether to stop when total reward does not increase for {patience} steps
params['patience']=1000         #Number of episodes where we allow a lower total reward before early stopping

params['load_model'] = False    #Wether to load actor and critic from a previous session and continue training
params['load_model_name'] = ""

#-------------Other-----------------------
params['make_debug_plots'] = True #Wether to plot statistics of gradients and train curve

with open('results/logs/{0}_config.csv'.format(params['savename']), 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    for key,value in env_params.items():
        writer.writerow([key,value])
    for key,value in params.items():
        writer.writerow([key,value])

if __name__=='__main__':
    DDPG = DDPG_agent(env,params)
    DDPG.train(run_async=False)
    #DDPG.train(run_async=not env_params['show_image'])
