import numpy as np
import matplotlib.pyplot as plt
from hcipy import *
import os
from collections import deque
import time
import matplotlib.pyplot as plt
np.seterr(divide='ignore',invalid='ignore')


class AO_env():
    def __init__(self,env_params):

        #-------------Set environment parameters (see main for description)--------
        self.wavelength = env_params['wavelength']
        self.D = env_params['D']
        self.pupil_pixels = env_params['pupil_pixels']
        self.show_image = env_params['show_image']
        self.verbosity = env_params['verbosity']
        self.turbulence_mode = env_params['turbulence_mode']
        self.servo_lag = env_params['servo_lag']
        self.num_airy = env_params['num_airy']
        self.num_photons = env_params['num_photons']
        self.reward_function = env_params['reward_function']
        self.num_iterations = env_params['num_iterations']
        self.control_mode = env_params['control_mode']
        self.closed_loop_freq = env_params['closed_loop_freq']
        self.temp_oversampling = env_params['temp_oversampling']
        self.wfs_error = env_params['wfs_error']
        self.leakage = 0.00
        self.oversize_pupil = 1.03
        self.dt = 1./(self.temp_oversampling*self.closed_loop_freq)

        #--------------Initialize pupil plane and non-aberrated wavefront------------------
        print('Initializing pupil plane')
        self.pupil_grid = make_pupil_grid(self.pupil_pixels,self.oversize_pupil*self.D)
        self.mag_grid = make_pupil_grid(self.pupil_pixels,self.oversize_pupil*self.D/1000)
        self.aperture = circular_aperture(self.D)(self.pupil_grid)
        self.input_wf = Wavefront(self.aperture,self.wavelength)
        self.input_wf.total_power = 1

        #--------------Initialize focal plane and propagator------------------------
        print('Initializing focal plane')
        q = 4 #Oversampling factor
        self.focal_pixels = 2*q*self.num_airy
        focal_plane = make_focal_grid_from_pupil_grid(self.pupil_grid,q=q,num_airy=self.num_airy,wavelength=self.wavelength)
        self.focal_camera = NoiselessDetector()
        self.coro_camera = NoiselessDetector()
        #self.noisy_focal_camera = NoisyDetector(focal_plane,read_noise=env_params['read_noise'],\
        #                                include_photon_noise=env_params['include_photon_noise'])

        self.pupil_to_focal = FraunhoferPropagator(self.pupil_grid,focal_plane)

        self.focal_camera.integrate(self.pupil_to_focal(self.input_wf),1./self.closed_loop_freq)
        self.Inorm = self.focal_camera.read_out().max()
        
        #---------Initialize list for DM commands------------------------
        self.action_wait_list = deque()

        #------------------------Create DM----------------------------
        print('Creating DM')
        self.num_act = env_params['num_actuators']
        #dm,self.num_modes,dm_modes,self.actuator_mask = make_actuator_DM(num_act=self.num_act\
        #        ,D=self.D,pupil_grid=self.pupil_grid)
        dm_modes = make_xinetics_influence_functions(self.pupil_grid,self.num_act,self.oversize_pupil*self.D/self.num_act)

        x = np.linspace(-self.oversize_pupil,self.oversize_pupil,self.num_act)
        XX, YY = np.meshgrid(x,-1*x)
        r_grid = np.sqrt(XX**2+YY**2)
        self.actuator_mask = (r_grid<=1.).ravel()

        self.A = dm_modes.transformation_matrix
        masked_A = np.asarray(self.A.todense())
        #masked_A = np.asarray(self.A.todense())*np.asarray(self.actuator_mask).ravel()
        A_in_ap = (masked_A.T*np.asarray(self.aperture)).T
        #self.A_inv = np.linalg.pinv(A_in_ap,rcond=8e-2) 
        self.A_inv = inverse_tikhonov(A_in_ap,rcond=1e-1)
        #self.A_inv = np.linalg.pinv(A_in_ap,rcond=9e-2)
        #self.A_inv = np.linalg.pinv(self.A.todense(),rcond=1e-1)
        self.num_modes = self.A.shape[1]


        #------------------------Create Coronagraph---------------------
        num_act_eff = self.num_act/self.oversize_pupil
        max_k = num_act_eff/2.4
        self.dark_zone = (np.abs(focal_plane.x)<(max_k*(self.wavelength/self.D)))*\
                        (np.abs(focal_plane.y)<(max_k*(self.wavelength/self.D)))

        #plt.imshow(self.dark_zone.reshape(self.focal_pixels,self.focal_pixels))
        #plt.show()
        #contrast = focal_plane.ones()
        #contrast[self.dark_zone] = contrast_level

        num_iterations = 100
        #self.vAPP = generate_app_keller(self.input_wf,self.pupil_to_focal,contrast,num_iterations,beta=1)
        self.coro = PerfectCoronagraph(self.aperture)
        
        #------------------------Create Atmosphere----------------------------
        self.L0 = env_params['L0']
        self.Cn2 = env_params['Cn2']
        self.heights = env_params['heights']
        self.angles = env_params['angles']
        self.velocity = env_params['velocity']
        self.t = 0

    def step(self,action=None):
        '''
        Iterates the AO system for one timestep.
        Arguments:
            action: Additional DM commands provided by the control algorithm
        Returns: 
            s: New state of the system
            r: Reward obtained
            terminate: Wether to terminate the episode or not
        '''
        self.iteration += 1
        if not action is None:
            action = action.ravel()
            action -= np.mean(action)
            self.action_wait_list.append((self.t-1./(2*self.closed_loop_freq),action.ravel()))
        
        wfs_phase = np.zeros_like(self.dm_phase)
        for k in range(self.temp_oversampling):
            self.t += self.dt
            if len(self.action_wait_list)>0:
                if self.t>=self.action_wait_list[0][0]+self.servo_lag:
                    self.dm_amps += self.action_wait_list[0][1] - self.leakage*self.dm_amps
                    self.action_wait_list.popleft()

                    #Calculate residual phase profile
                    self.dm_phase = self.A.dot(self.dm_amps)*self.aperture  #Multiply interaction matrix with DM amplitudes

            if self.turbulence_mode in ['atmosphere','atmosphere_random']:
                self.atmosphere.evolve_until(self.t)
                input_phase_screen = self.atmosphere.phase_for(self.wavelength)*self.aperture
            elif self.turbulence_mode=='dm':
                input_phase_screen = self.dm_modes_phase_screen
            self.res_phase = (input_phase_screen - self.dm_phase)
            self.res_phase -= np.mean(self.res_phase[self.aperture==1])
            self.res_phase = self.res_phase*self.aperture

            res_wf = self.input_wf.copy()
            res_wf.electric_field = self.aperture*res_wf.electric_field* np.exp(1j*self.res_phase)
            
            #Integrate focal plane image
            self.focal_camera.integrate(self.pupil_to_focal(res_wf),self.dt)

            #vAPP_wf = self.input_wf.copy()
            #vAPP_wf.electric_field = res_wf.electric_field*self.vAPP.electric_field
            #self.coro_camera.integrate(self.pupil_to_focal(vAPP_wf),self.dt)
            self.coro_camera.integrate(self.pupil_to_focal(self.coro(res_wf)),self.dt)
            wfs_phase += 1./self.temp_oversampling* self.res_phase 

        #if self.wfs_error>0:
        #    wfs_phase += self.wfs_error*np.random.randn(*wfs_phase.shape)
        #    wfs_phase += self.wfs_error*wfs_phase*np.random.randn(*wfs_phase.shape)
        #Calculate strehl, contrast and reconstruct WFS measurements
        focal_image = self.focal_camera.read_out()
        coro_image = self.coro_camera.read_out()
        s = np.array(self.A_inv.dot(wfs_phase))
        if self.wfs_error>0:
            s += self.wfs_error*np.random.randn(*s.shape)*self.actuator_mask.ravel()
        s = s.reshape(self.num_act,self.num_act,1)
        self.vapp_strehl = coro_image.max()
        self.strehl = focal_image.max()/self.Inorm
        self.contrast = np.std(coro_image[self.dark_zone])/self.Inorm

        if self.iteration>100:
            #self.science_image += coro_image.reshape(self.focal_pixels,self.focal_pixels)
            #self.science_image += focal_image/(self.Inorm*(self.num_iterations-51))
            self.science_image += focal_image/(self.Inorm*(self.num_iterations-101))
            self.science_coro_image += coro_image/(self.Inorm*(self.num_iterations-101))
            #self.science_camera.integrate(focal_image,self.dt/(self.num_iterations-51))
            self.phase_screens[self.iteration-101] = (self.res_phase*self.aperture).reshape(self.pupil_pixels,self.pupil_pixels)
            self.dm_shapes[self.iteration-101] = (self.dm_phase*self.aperture).reshape(self.pupil_pixels,self.pupil_pixels)
        r = self.reward_function(self.strehl,self.contrast,s)
        terminate = False
        if self.verbosity: print(int(self.t*self.closed_loop_freq),': Reward: {0:.3f}, Strehl: {1:.3f}, \
                Contrast: {2:.3e}'.format(np.mean(r),self.strehl,self.contrast),end='\r')

        if self.show_image:
            plt.figure(1,figsize=(12,8))
            plt.clf()
            plt.subplot(2,3,1)
            #plt.imshow(np.log10(np.array(focal_image)).reshape(self.focal_pixels,self.focal_pixels),vmin=-5)
            imshow_field(np.log10(focal_image/focal_image.max()),vmin=-5)
            plt.colorbar()
            plt.title('Focal plane')

            plt.subplot(2,3,2)
            imshow_field(np.log10(coro_image/coro_image.max()),vmin=-5)
            #plt.imshow(noisy_image.reshape(pixels,pixels))
            plt.title('Coronagraphic image')
            plt.colorbar()

            plt.subplot(2,3,3)
            plt.title('Input wavefront')
            #plt.imshow(input_phase_screen.reshape(self.pupil_pixels,self.pupil_pixels),cmap='bwr')
            plt.imshow(s[:,:,0].reshape(self.num_act,self.num_act),cmap='bwr')
            plt.colorbar()

            plt.subplot(2,3,4)
            plt.title('Action')
            #plt.imshow(s.reshape(self.pupil_pixels,self.pupil_pixels),cmap='bwr')
            plt.imshow(action.reshape(self.num_act,self.num_act),cmap='bwr')
            plt.colorbar()

            plt.subplot(2,3,5)
            plt.title('DM shape')
            plt.imshow(self.dm_phase.reshape(self.pupil_pixels,self.pupil_pixels),cmap='bwr')
            #plt.imshow(s[:,:,1].reshape(self.num_act,self.num_act),cmap='bwr')
            plt.colorbar()

            plt.subplot(2,3,6)
            plt.title('Residual wavefront')
            max_phase = np.std((self.res_phase))
            plt.imshow(self.res_phase.reshape(self.pupil_pixels,self.pupil_pixels),cmap='bwr',vmin=-2*max_phase,vmax=2*max_phase)
            #plt.imshow(s.reshape(self.num_act,self.num_act),cmap='bwr')
            plt.colorbar()

            plt.draw()
            plt.pause(0.1)
        return s,r,terminate


    def reset(self,reset_turbulence=False,reset_dm=False,input_amps=None):
        '''
        Resets the environment after an episode
        Arguments:
            reset_turbulence: Wether to generate new turbulence profile or not
            reset_dm: Wether to reset the DM amplitudes or not
            input_amps: When using the turbulence mode 'dm' this sets the modal coefficients of the turbulence
        Returns:
            s: Initial state of the system
        '''
        #Reset DM and DM command list
        self.science_image = np.zeros(self.focal_pixels*self.focal_pixels)
        self.science_coro_image = np.zeros(self.focal_pixels*self.focal_pixels)
        self.iteration = 0
        self.phase_screens = np.zeros((self.num_iterations-49,self.pupil_pixels,self.pupil_pixels))
        self.dm_shapes= np.zeros((self.num_iterations-49,self.pupil_pixels,self.pupil_pixels))
        self.dm_amps = np.zeros(self.num_modes)
        self.dm_phase = np.zeros(self.pupil_pixels**2)
        self.action_wait_list.clear()
        
        #Reset input turbulence
        self.t = 0 
        if self.turbulence_mode=='dm':
            initial_rms = np.random.uniform(0,1.5)
            if input_amps is None:
                random_amps = initial_rms*np.random.randn(self.num_act,self.num_act)
                self.dm_modes_phase_screen = self.A.dot(random_amps.ravel())
            else:
                self.dm_modes_phase_screen = self.A.dot(input_amps*self.actuator_mask)

        if self.turbulence_mode=='atmosphere':
            layers = []
            for (velocity,angle,Cn2,L0,height) in zip(self.velocity,self.angles,self.Cn2,\
                    self.L0,self.heights):
                wind_vector = velocity*np.array([np.cos(angle/180*np.pi),np.sin(angle/180*np.pi)])
                layers.append(InfiniteAtmosphericLayer(self.pupil_grid,
                    Cn2,L0,wind_vector,height,2,use_interpolation=True))
            self.atmosphere = MultiLayerAtmosphere(layers,scintillation=False)

        elif self.turbulence_mode=='atmosphere_random':
            layers = []
            ground_angle = np.random.uniform(0,2*np.pi)
            ground_velocity = np.random.uniform(5,12)
            jet_angle = np.random.uniform(0,2*np.pi)
            jet_velocity = np.random.uniform(10,35)
            print('Velocity: {0:.1f}, Angle: {1:.1f}'.format(jet_velocity,jet_angle/np.pi*180))

            ground_vector = ground_velocity * np.array([np.cos(ground_angle),np.sin(ground_angle)])
            jet_vector = jet_velocity * np.array([np.cos(jet_angle),np.sin(jet_angle)])

            layers.append(InfiniteAtmosphericLayer(self.pupil_grid,
                    self.Cn2[0],self.L0[0],ground_vector,self.heights[0],2,use_interpolation=True))
            layers.append(InfiniteAtmosphericLayer(self.pupil_grid,
                    self.Cn2[1],self.L0[1],jet_vector,self.heights[1],2,use_interpolation=True))

            self.atmosphere = MultiLayerAtmosphere(layers,scintilation=False)

        if self.control_mode=='state':
            s,r,terminate = self.step(np.zeros(2*self.N_mla*self.N_mla))
        else:
            s,r,terminate = self.step(np.zeros(self.num_modes))

        self.t = 0
        return(s)

def make_actuator_DM(num_act,pupil_grid,D=1):
    '''
    Function to make deformable mirror modes with gaussian response
    Arguments:
        num_act: Number of actuators across pupil diameter
        pupil_grid: 
        D: diameter of the pupil
    Returns:
        dm: hcipy Deformable Mirror object
        num_modes: Number of mirror modes
        dm_modes: Mode basis
        actuator_mask: Mask of actuators that are within illuminated pupil
    '''
    act_distance = D/num_act
    x = np.linspace(-1.1*D/2,1.1*D/2,num_act)
    XX, YY = np.meshgrid(x,-1*x)
    r_grid = np.sqrt(XX**2+YY**2)
    actuator_mask = r_grid<=1.05*(D/2)
    x = XX[actuator_mask].ravel()
    y = YY[actuator_mask].ravel()
    actuator_grid = CartesianGrid(UnstructuredCoords([x,y]))
    num_modes = np.sum(actuator_mask)
    print('\n\n\n Number of modes:',num_modes)
    time.sleep(2)
    pitch = D/num_act
    dm_modes = make_gaussian_pokes(pupil_grid,actuator_grid,pitch/np.sqrt(2))
    dm = DeformableMirror(dm_modes)
    return(dm,num_modes,dm_modes,actuator_mask)

