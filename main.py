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
    #reward = np.log(strehl/(contrast/1e-5))
    reward = np.log(strehl)
    #reward = -(modal_res)**2
    #reward = -1*np.log10(contrast/1e-4)
    #reward = np.log10(vapp_strehl/(contrast/1e-5))
    #reward = -1*np.sqrt(np.mean(centers**2))
    #reward = strehl
    #reward = -1*np.log(centers**2)
    return reward

env_params = dict()
env_params['D'] = 4  #Diameter of simulated telescope
env_params['wavelength'] = 0.658e-6   #Wavelength (m) of monochromatic source
#env_params['wavelength'] = 0.532e-6   #Wavelength (m) of monochromatic source
env_params['reward_function'] = reward_function    #Reward function
env_params['pupil_pixels'] = 128  #Number of pixels in pupil_plane
env_params['num_iterations'] = 200  #Number of iterations per episode
env_params['show_image'] = False
#Visualize wavefronts for debugging of simulations, this is too slow for real training
env_params['verbosity'] = True      #Print progress
#Time (in iterations) between sensed wavefront and correction of the DM, only integers >0 are supported.
#Is this realistic? I calculate the Strehl every discrete step and not inbetween.
env_params['num_airy'] = 16 #Size of focal plane in number of airy rings
env_params['num_photons'] = np.inf #Number of photons in incoming beam for adding photon noise
env_params['wfs_error'] = 0.
env_params['closed_loop_freq'] = 1000   #Frequency (Hz) to run the correction
#env_params['closed_loop_freq'] = 1380   #Frequency (Hz) to run the correction
env_params['servo_lag'] = 0/env_params['closed_loop_freq']
env_params['temp_oversampling'] = 1

env_params['turbulence_mode'] = 'atmosphere'
#Turbulence mode, options:
#   dm: Turbulence only consists of combination of mirror modes
#   atmosphere: Atmospheric turbulence with fixed wind velocity and angle
#   atmosphere_random: Two-layered atmospheric turbulence with random wind velocity and angle for every episode

env_params['L0'] = [50]  #Outer scale parameter of simulated atmospheric layer
env_params['Cn2'] = [Cn_squared_from_fried_parameter(r0=0.2,wavelength=env_params['wavelength'])]
#env_params['Cn2'] = [0.8e-13,1.3e-13]
env_params['angles'] = [0]
env_params['velocity'] = [25]  #Wind velocity vector
#env_params['velocity'] = [12,30]  #Wind velocity vector
#env_params['velocity'] = [0,0]  #Wind velocity vector
#env_params['heights'] = [0,11e3]
env_params['heights'] = [0]

env_params['num_actuators'] = 25 #Number of actuators of the DM along pupil diameter
env_params['N_mla'] = 32 #Number of microlenses in Shack-Hartmann WFS

env_params['control_mode'] = 'modal'
#Control mode, options:
#   modal: Controller takes as input reconstructed modal coefficients and returns modal DM commands.
#   state: Controller takes as input slope measurements and returns slope measurements to correct for.
#   both: Controller takes as input slope measurements and returns modal DM commands.
#NOTE: If 'both' is used the architecture needs to be adjusted accordingly.

env_params['reconstruction_matrix_name'] =  'reconstruction_matrix_xinetics24.txt'
env_params['redo_calibration'] = not env_params['show_image']
#Filename of the reconstruction matrix (if modal or state control method is used)
#Reconstruction matrix R should have the form a = Rs, with a the modal coefficients and s the estimated slopes.
#If file does not exist, it will do the calibration and save the result

print('Initializing AO environment')
env = AO_env(env_params)

#--------------Hyperparameters of networks------------
#Architectures of networks can be changed in ddpg/actor.py and ddpg/critic.py.
params = dict()
params['actor_lr'] = 3e-5       #Learning rate for the actor
params['actor_lr_decay'] = np.exp(np.log(0.5)/500.)

params['critic_lr'] = 1e-3      #Learning rate for the critic
params['critic_lr_decay'] = np.exp(np.log(0.5)/500.)

params['tau'] = 3e-3      #Update parameter for the target networks: Q' = (1-tau)Q' + tau Q 
params['actor_grad_clip'] = 1.   #Clipping of the gradient for updating the actor to avoid large changes
params['use_stateful_actor'] = False
params['pretrain_actor'] = True
params['pretrain_gain'] = 0.3
#Wether to use a stateful actor as a controller. (https://fairyonice.github.io/Stateful-LSTM-model-training-in-Keras.html explains the difference)
#If True, the hidden state is preserved during an episode and only the most recent observation is propagated through the LSTM.
#If False, the input to the actor is a fixed number of timesteps with the hidden state initialized to zero.
#The latter sometimes gives more stable training but has less efficient online prediction.
#This is maybe because TBTT is only performed over a small number of timesteps so initialization is important 
#    -> solve it with variable length training input or longer sequences?

#--------------RL algorithm hyperparameters-----------
params['gamma'] = 0. #Discount factor for expected future rewards
params['reward_type'] = 'focal'
params['buffer_size'] = 100 #Maximum number of episodes to save in the replay buffer
params['minibatch_size'] = 1 #Batch size for training critic and actor
params['num_training_batches'] = 200  #Number of batches to train on every episode
params['optimization_length'] = 2 #Size of the history/number of steps to use in the BPTT
params['initialization_length'] = 1
params['trajectory_length'] = 1     #Number of steps to use observed rewards instead of bootstrapping with target critic.
#This does not work properly at ends op episodes and should be 1.

params['warmup'] = 0
params['actor_warmup'] = 3
params['action_scaling'] = 3     #Maximum possible action
params['use_integrator'] = False
params['integrator_gain'] = 0.5
#Probability of randomly using integrator for an iteration to fill the replay buffer with 'good' experience.
#This can be used to improve training stability. 
#Gain of the integrator is defaulted to 0.3 and can be set in ddpg/ddpg_agent.py

#-------------Closed Loop parameters------------------
params['iterations_per_episode'] = env_params['num_iterations']

#-------------Exploration noise parameters----------
params['start_noise'] = 0.1 #standard deviation of the random action noise
#params['noise_type'] = 'gaussian'       #Action noise type: gaussian or ou
params['theta_noise'] = 0.8             #Only used for ou noise
params['noise_decay'] = np.exp(np.log(0.5)/1000.)            #noise decay factor
params['noise_type'] = 'action'    #wether to use action or parameter space noise for exploration

#---------------Save/load parameters-------------------
params['max_episodes']= 1000            #Number of episodes to train for

params['savename'] = "fully_modal_linear"
#params['savename'] = "test"

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
    #DDPG.train(run_async=True)
    DDPG.train(run_async=not env_params['show_image'])
