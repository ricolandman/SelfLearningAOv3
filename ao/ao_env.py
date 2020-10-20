import numpy as np
import matplotlib.pyplot as plt
hcipy_version='new'
if hcipy_version=='new':
    from hcipy import *
    from hcipy.optics.magnifier import Magnifier
else:
    from hcipy_old import *
    from hcipy_old.optics.magnifier import MonochromaticMagnifier
import os
from collections import deque
import time
from ao.sh_reconstructor import *
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

        #--------------Initialize pupil plane and non-aberrated wavefront------------------
        print('Initializing pupil plane')
        self.pupil_grid = make_pupil_grid(self.pupil_pixels,self.D)
        self.mag_grid = make_pupil_grid(self.pupil_pixels,self.D/1000)
        self.aperture = circular_aperture(self.D)(self.pupil_grid)
        self.input_wf = Wavefront(self.aperture,self.wavelength)
        self.input_wf.total_power = 1

        #--------------Initialize focal plane and propagator------------------------
        print('Initializing focal plane')
        q = 8 #Oversampling factor
        self.focal_pixels = 2*q*self.num_airy
        focal_plane = make_focal_grid_from_pupil_grid(self.pupil_grid,q=q,num_airy=self.num_airy,wavelength=self.wavelength)
        if hcipy_version=='new':
            self.pupil_to_focal = FraunhoferPropagator(self.pupil_grid,focal_plane)
        else:
            self.pupil_to_focal = FraunhoferPropagator(self.pupil_grid,focal_plane,self.wavelength)
        wf_out = self.pupil_to_focal(self.input_wf)
        image = wf_out.intensity.reshape(self.focal_pixels,self.focal_pixels)
        self.Inorm = self.pupil_to_focal(self.input_wf).intensity.max()

        
        #---------Initialize list for DM commands------------------------
        self.action_wait_list = deque()

        #------------------------Create WFS----------------------------
        print('Creating WFS')
        if env_params['wavefront_sensor'] == 'shack-hartmann':
            self.N_mla = env_params['N_mla']
            self.F_mla = 32
            self.magnification = 1000
            if hcipy_version=='new':
                self.magnifier = Magnifier(1./self.magnification) #Magnify incoming beam by factor 1000
            else:
                self.magnifier = MonochromaticMagnifier(1./self.magnification,self.wavelength) #Magnify incoming beam by factor 1000

            mla_diameter = float(self.D)/self.magnification
            lenslet_diameter = mla_diameter/self.N_mla
            x = np.arange(-mla_diameter/2+0.5*lenslet_diameter, mla_diameter/2, lenslet_diameter)
            mla_grid = CartesianGrid(SeparatedCoords((x, x)))
            self.focal_length = self.F_mla*lenslet_diameter
            micro_lens_array = MicroLensArray(self.mag_grid,mla_grid,self.focal_length)

            self.wfs = ShackHartmannWavefrontSensorOptics(self.mag_grid,micro_lens_array)
            self.wfs_estimator = ShackHartmannWavefrontSensorEstimator(mla_grid,\
                    micro_lens_array.mla_index)

            ref_image = self.wfs(self.magnifier(self.input_wf)).intensity
            self.ref_centers = self.wfs_estimator.estimate([ref_image])
            time.sleep(2)
            
            x = np.linspace(-1,1,self.N_mla)
            XX, YY = np.meshgrid(x,-1*x)
            r_grid = np.sqrt(XX**2+YY**2)
            self.spot_mask = (r_grid<1.01)
        elif env_params['wavefront_sensor']=='perfect':
            self.N_mla = self.pupil_pixels

        #------------------------Create DM----------------------------
        print('Creating DM')
        self.num_act = env_params['num_actuators']
        dm,self.num_modes,dm_modes,self.actuator_mask = make_actuator_DM(num_act=self.num_act\
                ,D=self.D,pupil_grid=self.pupil_grid)
        self.A = dm_modes.transformation_matrix


        #------------------------Create Coronagraph---------------------
        contrast_level = 1e-5
        self.dark_zone = (circular_aperture(100*focal_plane.delta[0])(focal_plane)).astype(bool)*(focal_plane.x>2*focal_plane.delta[0])

        contrast = focal_plane.ones()
        contrast[self.dark_zone] = contrast_level

        num_iterations = 80
        self.vAPP = generate_app_keller(self.input_wf,self.pupil_to_focal,contrast,num_iterations,beta=1)
        
        #------------------------Create Atmosphere----------------------------
        self.L0 = env_params['L0']
        self.Cn2 = env_params['Cn2']
        self.heights = env_params['heights']
        self.angles = env_params['angles']
        self.velocity = env_params['velocity']
        self.t = 0
        
        #-------------Load calibration matrix------------------------------
        #If the reconstruction matrix does not exist, we need to do the calibration
        if self.control_mode in ['modal','state']:
            if not os.path.exists('./models/'+env_params['reconstruction_matrix_name']):
                print('Calibrating reconstruction matrix...')
                #Temporarily set env params for calibration
                self.turbulence_mode='dm'
                self.num_photons = np.inf
                self.control_mode = ['both']  #We want to input modal coefficients and get slope measurements

                #Calibrate reconstruction matrix
                self.reconstructor = Reconstructor()
                #self.reconstructor.calibrate(self,trials=10000,cal_rms=0.5,rconds=np.logspace(0,-4,20),plot=False)
                self.reconstructor.calibrate2(self,cal_rms=1,rconds=np.logspace(-1,-5,10))
                self.reconstructor.save(env_params['reconstruction_matrix_name'])

                #Reset env params to original values
                self.turbulence_mode = env_params['turbulence_mode']
                self.num_photons = env_params['num_photons']
                self.control_mode = env_params['control_mode']

            else:
                self.reconstructor = Reconstructor()
                self.reconstructor.load(env_params['reconstruction_matrix_name'])

    def step(self,action):
        '''
        Iterates the AO system for one timestep.
        Arguments:
            action: Additional DM commands provided by the control algorithm
        Returns: 
            s: New state of the system
            r: Reward obtained
            terminate: Wether to terminate the episode or not
        '''
        #Get new phase screen
        if len(action.shape)>1:
            action = action[self.actuator_mask]
        self.t += 1./self.closed_loop_freq
        if self.turbulence_mode in ['atmosphere','atmosphere_random']:
            self.atmosphere.evolve_until(self.t)
            input_phase_screen = self.atmosphere.phase_for(self.wavelength)*self.aperture
        elif self.turbulence_mode=='dm':
            input_phase_screen = self.dm_modes_phase_screen

        #Apply DM commands
        if self.control_mode=='state':
            action = self.reconstructor.reconstruct(action.reshape(1,-1))
        #masked_action = action.reshape(self.num_act,self.num_act)*self.actuator_mask
        self.action_wait_list.append(action.ravel())
        self.dm_amps += self.action_wait_list[0]
        self.action_wait_list.popleft()

        #Calculate residual phase profile
        dm_phase = self.A.dot(self.dm_amps)  #Multiply interaction matrix with DM amplitudes
        self.res_phase = (input_phase_screen - dm_phase)
        self.res_phase -= np.mean(self.res_phase[self.aperture==1])
        self.res_phase = self.res_phase*self.aperture

        #Get image from wfs
        res_wf = self.input_wf.copy()
        res_wf.electric_field = self.aperture*res_wf.electric_field* np.exp(1j*self.res_phase)
        mag_wf = self.magnifier(res_wf)
        wf_out = self.wfs(mag_wf)
        image = wf_out.intensity

        #Add noise to image
        image = image/np.sum(image)
        noisy_image = image
        if self.num_photons is not np.inf:
            noisy_image[:] = np.random.poisson(self.num_photons*image)
        noisy_image = noisy_image/np.sum(noisy_image)
        
        #Estimate spot centers
        o = self.wfs_estimator.estimate([noisy_image])-self.ref_centers
        o = np.nan_to_num(o)

        #Change shape of centers and convert to units of radians
        o = np.flip(o.transpose().reshape(self.N_mla,self.N_mla,2),axis=1)*\
                (self.magnification/self.focal_length)
        o = (o.transpose()*self.spot_mask).transpose()
        
        #Reconstruct amplitudes
        if self.control_mode=='modal':
            #s = np.expand_dims(o.ravel(),axis=0)
            #s = self.reconstructor.reconstruct(s).reshape(self.num_act,self.num_act,1)*\
            #        self.actuator_mask.reshape(self.num_act,self.num_act,1)
            s = np.zeros((self.num_act,self.num_act))
            s[self.actuator_mask] = self.reconstructor.reconstruct(o.ravel())
            s = np.expand_dims(s,axis=-1)
        else:
            s=o


        #Calculate Strehl and reward
        focal_image = self.pupil_to_focal(res_wf).intensity/self.Inorm
        vAPP_wf = self.input_wf.copy()
        vAPP_wf.electric_field = res_wf.electric_field*self.vAPP.electric_field
        coro_image = self.pupil_to_focal(vAPP_wf).intensity/self.Inorm
        self.contrast = np.mean(coro_image[self.dark_zone])
        self.strehl = focal_image.max()
        self.strehl_vapp = coro_image.max()
        wavefront_rms = np.std(self.res_phase[self.aperture==1])
        spot_rms = np.sqrt(np.mean(np.sqrt(np.sum(o[self.spot_mask]**2,axis=1))))
        #modal_rms = np.sqrt(np.mean(s[self.actuator_mask]**2))
        modal_rms = spot_rms
        #x_rms = np.sqrt(np.mean(o[self.spot_mask][:,0]**2))
        #y_rms = np.sqrt(np.mean(o[self.spot_mask][:,1]**2))
        #spot_rms = x_rms**2+y_rms**2
        r = self.reward_function(self.strehl,self.strehl_vapp,self.contrast)
        terminate = False
        if self.verbosity: print(int(self.t*self.closed_loop_freq),': Reward: {0:.3f}, Strehl: {1:.3f},\
                Contrast: {2:.3e}'.format(np.sum(r),self.strehl,self.contrast),end='\r')

        if self.show_image:
            plt.clf()
            plt.subplot(2,2,1)
            plt.imshow(np.log10(np.array(focal_image)).reshape(self.focal_pixels,self.focal_pixels),vmin=-5)
            plt.colorbar()
            plt.title('Focal plane')
            plt.subplot(2,2,2)
            pixels = int(np.sqrt(noisy_image.size))
            plt.imshow(np.log10(coro_image.reshape(self.focal_pixels,self.focal_pixels)),vmin=-5)
            #plt.imshow(noisy_image.reshape(pixels,pixels))
            plt.title('WFS measurement')
            plt.colorbar()
            plt.subplot(2,2,3)
            plt.title('Input wavefront')
            #plt.imshow(input_phase_screen.reshape(self.pupil_pixels,self.pupil_pixels),cmap='bwr')
            plt.imshow(o[:,:,0],cmap='bwr')
            plt.colorbar()
            plt.subplot(2,2,4)
            plt.title('Residual wavefront')
            plt.imshow(self.res_phase.reshape(self.pupil_pixels,self.pupil_pixels),cmap='bwr')
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
        self.dm_amps = np.zeros(self.num_modes)
        self.action_wait_list.clear()
        for _ in range(self.servo_lag-1):
            self.action_wait_list.append(np.zeros(self.num_modes))
        
        #Reset input turbulence
        self.t = 0 
        if self.turbulence_mode=='dm':
            initial_rms = np.random.uniform(0,1.5)
            if input_amps is None:
                random_amps = initial_rms*np.random.randn(self.num_act,self.num_act)*self.actuator_mask
                random_amps -= np.mean(random_amps[self.actuator_mask])
                self.dm_modes_phase_screen = self.A.dot(random_amps.ravel())
            else:
                self.dm_modes_phase_screen = self.A.dot(input_amps)

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
    x = np.linspace(-D/2,D/2,num_act)
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

