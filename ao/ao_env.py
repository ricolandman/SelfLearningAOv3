import numpy as np
import matplotlib.pyplot as plt
from hcipy import *
import os
from collections import deque
import time
import scipy.ndimage as ndimage
from .custom_pyramid import *
import matplotlib.pyplot as plt
np.seterr(divide='ignore',invalid='ignore')


class AO_env():
    def __init__(self,env_params):

        #-------------Set environment parameters (see main for description)--------
        self.wavelength_science = env_params['wavelength_science']
        self.wavelength_wfs = env_params['wavelength_sensing']
        self.D = env_params['D']
        self.pupil_pixels = env_params['pupil_pixels']
        self.show_image = env_params['show_image']
        self.verbosity = env_params['verbosity']
        self.turbulence_mode = env_params['turbulence_mode']
        self.servo_lag = env_params['servo_lag']
        self.num_airy = env_params['num_airy']
        self.num_photons = env_params['num_photons']
        zero_magnitude_flux = 3.9e10
        self.num_photons = zero_magnitude_flux * 10**(-env_params['stellar_magnitude']/ 2.5)
        self.reward_function = env_params['reward_function']
        self.num_iterations = env_params['num_iterations']
        self.burnin_iterations = env_params['burnin_iterations']
        self.control_mode = env_params['control_mode']
        self.closed_loop_freq = env_params['closed_loop_freq']
        self.temp_oversampling = env_params['temp_oversampling']
        self.leakage = 0.00
        self.dt = 1./(self.temp_oversampling*self.closed_loop_freq)

        #--------------Initialize pupil plane and non-aberrated wavefront------------------
        print('Initializing pupil plane')
        central_obscuration = 1.2*self.D/8
        central_obscuration_ratio = central_obscuration / self.D
        spider_width = 0.05*self.D/8 # meter
        self.oversize_pupil = 16 / 15
        self.pupil_pixels = int(self.pupil_pixels*self.oversize_pupil)
        pupil_grid_diameter = self.D * self.oversize_pupil
        self.pupil_grid = make_pupil_grid(self.pupil_pixels, pupil_grid_diameter)

        VLT_aperture_generator = make_obstructed_circular_aperture(self.D, 
            central_obscuration_ratio, num_spiders=4, spider_width=spider_width)

        self.aperture = evaluate_supersampled(VLT_aperture_generator, self.pupil_grid, 4)

        self.input_wf = Wavefront(self.aperture, self.wavelength_science)
        self.input_wf.total_power = self.num_photons

        self.input_wf_wfs = Wavefront(self.aperture, self.wavelength_wfs)
        self.input_wf_wfs.total_power = self.num_photons

        #--------------Initialize focal plane and propagator------------------------
        print('Initializing focal plane')
        q = 4 #Oversampling factor
        spatial_resolution = self.wavelength_science/self.D
        focal_grid = make_focal_grid(q=q, num_airy=self.num_airy, spatial_resolution=spatial_resolution)
        self.focal_pixels = 2*q*self.num_airy
        self.focal_camera = NoiselessDetector()
        self.coro_camera = NoiselessDetector()
        #self.noisy_focal_camera = NoisyDetector(focal_grid,read_noise=env_params['read_noise'],\
        #                                include_photon_noise=env_params['include_photon_noise'])

        self.prop = FraunhoferPropagator(self.pupil_grid, focal_grid)

        self.focal_camera.integrate(self.prop(self.input_wf),1./self.closed_loop_freq)
        unaberrated_psf = self.focal_camera.read_out()
        self.Inorm = unaberrated_psf.max()
        
        #---------Initialize list for DM commands------------------------
        self.action_wait_list = deque()

        #------------------------Create DM----------------------------
        print('Creating DM')
        self.num_act = env_params['num_actuators']
        self.dm, self.actuator_mask = make_actuator_dm(self.pupil_grid, self.num_act, self.D*self.oversize_pupil)

        self.num_modes = self.dm.num_actuators
        
        
        #------------------------Create Coronagraph---------------------
        num_act_eff = self.num_act/self.oversize_pupil
        max_k = num_act_eff/2.4
        self.dark_zone = (np.abs(focal_grid.x)<(max_k*(self.wavelength_science/self.D)))*\
                        (np.abs(focal_grid.y)<(max_k*(self.wavelength_science/self.D)))

        #plt.imshow(self.dark_zone.reshape(self.focal_pixels,self.focal_pixels))
        #plt.show()
        #contrast = focal_grid.ones()
        #contrast[self.dark_zone] = contrast_level

        #self.vAPP = generate_app_keller(self.input_wf,self.prop,contrast,num_iterations,beta=1)
        self.coro = PerfectCoronagraph(self.aperture, 4)

        #---------------------Create WFS-------------------------------
        print('Creating WFS')
        self.wfs_type = env_params['wfs_type']
        self.wfs_camera = NoiselessDetector()
        if self.wfs_type=='phase':
            #if self.scintillation:
            #    raise ValueError('Cannot use direct phase sensor when including scintillation')
            transformation_matrix = np.asarray(self.A.todense())
            masked_transformation_matrix = (transformation_matrix.T*np.asarray(self.aperture)).T
            self.reconstruction_matrix = inverse_tikhonov(masked_transformation_matrix,rcond=1e-2)
            #self.wfs = lambda x: x.phase
            self.wfs = lambda x: Field(np.unwrap((x.phase*self.aperture).shaped, discont=1.8*np.pi).ravel(),self.pupil_grid)*self.aperture
            self.get_wfs_measurement = lambda x: x*self.closed_loop_freq
            #self.get_wfs_measurement = lambda x: Field(np.unwrap(x.shaped*self.closed_loop_freq).ravel(),self.pupil_grid)

        else:
            if self.wfs_type=='pyramid':
                self.wfs = CustomPyramidWavefrontSensorOptics(self.pupil_grid, num_output_pixels=self.pupil_pixels, wavelength_0=self.wavelength_wfs)
                clean_wfs_measurement = lambda x: x/np.sum(x)
                N_wfs = self.pupil_pixels**2
                #sub_shape = self.ref_image.shaped.shape//2
                #clean_wfs_measurement = lambda x: np.array([x[:sub_shape[0], :sub_shape[1]], \
                #                        x[sub_shape[0]:, :sub_shape[1]], \
                #                        x[:sub_shape[0], sub_shape[1]:], \
                #                        x[sub_shape[0]:, sub_shape[1]:] ])/np.sum(x)

            elif self.wfs_type=='shack-hartmann':
                #focal_length = 1
                #self.N_mla = np.ceil(self.oversize_pupil*env_params['N_mla'])
                f_number = 50
                sh_diameter = 5e-3
                magnification = sh_diameter/self.D
                mag = Magnifier(magnification)
                shwfs = SquareShackHartmannWavefrontSensorOptics(self.pupil_grid.scaled(magnification),\
                                         f_number, env_params['N_mla'], sh_diameter)
                shwfse = ShackHartmannWavefrontSensorEstimator(shwfs.mla_grid, shwfs.micro_lens_array.mla_index)
                self.wfs = lambda x: shwfs(mag(x)).power

                #Get reference image
                image_ref = self.wfs(self.input_wf_wfs)

                #Only use subapertures with more than 50% of max subaperture flux
                fluxes = ndimage.measurements.sum(image_ref, shwfse.mla_index, shwfse.estimation_subapertures)
                flux_limit = fluxes.max() * 0.5
                estimation_subapertures = shwfs.mla_grid.zeros(dtype='bool')
                estimation_subapertures[shwfse.estimation_subapertures[fluxes > flux_limit]] = True
                N_wfs = 2* np.sum(estimation_subapertures)

                self.wfs_estimator = ShackHartmannWavefrontSensorEstimator(shwfs.mla_grid,\
                             shwfs.micro_lens_array.mla_index, estimation_subapertures)

                clean_wfs_measurement = lambda x: self.wfs_estimator.estimate([x+1e-10]).ravel()
            else:
                raise ValueError("Unknown WFS type {0}, options are: phase, shack-hartmann, pyramid".format(self.wfs_type))

            self.wfs_camera.integrate(self.wfs(self.input_wf_wfs),1./self.closed_loop_freq)
            ref_image = self.wfs_camera.read_out()
            self.ref_measurement = clean_wfs_measurement(ref_image)
            self.get_wfs_measurement = lambda x: clean_wfs_measurement(x)-self.ref_measurement

        #Calibrate WFS
        interaction_matrix_name = 'models/interaction_matrix_{0}.npy'.format(self.wfs_type)

        loaded_calibration_matrix = False
        if os.path.exists(interaction_matrix_name):
            self.reconstruction_matrix = np.load(interaction_matrix_name)
            print(self.reconstruction_matrix.shape)
            print(self.num_modes, N_wfs)
            if self.reconstruction_matrix.shape == (self.num_modes, N_wfs):
                loaded_calibration_matrix = True
                print('Loaded interaction matrix from ', interaction_matrix_name)

        if not loaded_calibration_matrix:
            print('Calibrating WFS')
            slopes = []
            probe_amp = 0.3
            start = time.time()
            for ind in range(self.num_modes):
                print("Measure response to mode {:d} / {:d}".format(ind+1, self.num_modes), end='\r')
                slope = 0

                # Probe the phase response
                for s in [1, -1]:
                    amp = np.zeros((self.num_modes,))
                    amp[ind] = s * probe_amp
                    self.dm.actuators = amp*self.wavelength_science/(2*np.pi)

                    dm_wf = self.dm.forward(self.input_wf_wfs)

                    self.wfs_camera.integrate(self.wfs(dm_wf), 1./self.closed_loop_freq)
                    image = self.wfs_camera.read_out()
                    #plt.clf()
                    #imshow_field(image, cmap='inferno')
                    #plt.colorbar()
                    #plt.draw()
                    #plt.pause(0.01)
                    measurement = self.get_wfs_measurement(image)
                    slope += s * measurement/(2 * probe_amp)

                slopes.append(slope)
            #plt.close()

            slopes = ModeBasis(slopes) 
            print('Inverting calibration matrix')
            self.reconstruction_matrix = inverse_tikhonov(slopes.transformation_matrix, rcond=1e-1, svd=None)
            np.save(interaction_matrix_name, self.reconstruction_matrix)
            print('Calibration took {0:.1f} seconds'.format(time.time()-start))

        
        #------------------------Create Atmosphere----------------------------
        self.L0 = env_params['L0']
        self.Cn2 = env_params['Cn2']
        self.heights = env_params['heights']
        self.angles = env_params['angles']
        self.velocity = env_params['velocity']
        self.scintillation = env_params['scintillation']
        self.t = 0

    def step(self,action=None, reconstruct = False):
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
            
            #Get 
            t_command = self.t-0.5*self.closed_loop_freq
            if not action.size==self.num_modes:
                action = action.ravel()[self.actuator_mask]
            self.action_wait_list.append((t_command, action))
            #self.action_wait_list.append((self.t-1./(2*self.closed_loop_freq),action.ravel()))
        
        wfs_phase = np.zeros_like(self.dm_phase)
        for k in range(self.temp_oversampling):
            self.t += self.dt
            if len(self.action_wait_list)>0:
                if self.t>=self.action_wait_list[0][0]+self.servo_lag:
                    self.dm_amps += self.action_wait_list[0][1] - self.leakage*self.dm_amps
                    self.dm.actuators = self.dm_amps*self.wavelength_wfs/(2*np.pi)
                    self.action_wait_list.popleft()

            self.atmosphere.evolve_until(self.t)
            atm_wf = self.atmosphere(self.input_wf)
            dm_wf = self.dm(atm_wf)
            res_phase = dm_wf.phase*self.aperture

            #Integrate cameras
            self.focal_camera.integrate(self.prop(dm_wf),self.dt)
            self.coro_camera.integrate(self.prop(self.coro(dm_wf)),self.dt)
            self.wfs_camera.integrate(self.wfs(self.dm(self.atmosphere(self.input_wf_wfs))),self.dt)
            #wfs_phase += 1./self.temp_oversampling*res_phase 

        #Calculate strehl, contrast and reconstruct WFS measurements
        focal_image = self.focal_camera.read_out()/self.Inorm
        self.strehl = focal_image.max()

        coro_image = self.coro_camera.read_out()/self.Inorm
        self.contrast = np.mean(coro_image[self.dark_zone])

        wfs_image = self.wfs_camera.read_out()
        wfs_image = large_poisson((wfs_image).astype('float'))
        o = self.get_wfs_measurement(wfs_image)
        amps = self.reconstruction_matrix.dot(o)
        s = np.zeros((self.num_act**2))
        s[self.actuator_mask] = amps
        s = s.reshape(self.num_act,self.num_act,1)
        wavefront_rms = np.sqrt(np.mean((wfs_phase-np.mean(wfs_phase))**2))

        if self.iteration>=self.burnin_iterations:
            #self.science_image += coro_image.reshape(self.focal_pixels,self.focal_pixels)
            #self.science_image += focal_image/(self.Inorm*(self.num_iterations-51))
            self.science_image += focal_image.shaped/(self.num_iterations-self.burnin_iterations)
            self.science_coro_image += coro_image.shaped/(self.num_iterations-self.burnin_iterations)
            #self.science_camera.integrate(focal_image,self.dt/(self.num_iterations-51))
            self.wavefront_variance += (res_phase.shaped-np.mean(res_phase[self.aperture==1]))**2/(self.num_iterations-self.burnin_iterations)
            self.mean_dm_shape += self.dm.surface.shaped/(self.num_iterations-self.burnin_iterations)
            self.mean_res_phase += (res_phase.shaped-np.mean(res_phase[self.aperture==1]))/(self.num_iterations-self.burnin_iterations)
            #self.phase_screens[self.iteration-self.burnin_iterations-1] = (res_phase*self.aperture).shaped
            #self.dm_shapes[self.iteration-self.burnin_iterations-1] = (self.dm_phase*self.aperture).shaped
        r = self.reward_function(self.strehl,self.contrast,s)
        terminate = False
        if self.verbosity: print(int(self.t*self.closed_loop_freq),': Reward: {0:.3f}, Wavefront RMS: {3:.2f}, Strehl: {1:.3f}, \
                Contrast: {2:.3e}'.format(np.mean(r),self.strehl,self.contrast, wavefront_rms),end='\r')

        if self.show_image:
            plt.figure(1,figsize=(12,8))
            plt.clf()
            plt.subplot(2,3,1)
            #plt.imshow(np.log10(np.array(focal_image)).reshape(self.focal_pixels,self.focal_pixels),vmin=-6)
            imshow_field(np.log10(focal_image),vmin=-5,vmax=0,cmap='inferno')
            plt.axis('off')
            plt.colorbar()
            plt.title('Focal plane')

            plt.subplot(2,3,2)
            imshow_field(np.log10(coro_image),vmin=-5,vmax=-2,cmap='inferno')
            plt.axis('off')
            #plt.imshow(noisy_image.reshape(pixels,pixels))
            plt.title('Coronagraphic focal plane')
            plt.colorbar()

            plt.subplot(2,3,3)
            plt.title('Wavefront Sensor')
            plt.axis('off')
            #plt.imshow(input_phase_screen.reshape(self.pupil_pixels,self.pupil_pixels),cmap='bwr')
            #plt.imshow(s[:,:,0].reshape(self.num_act,self.num_act),cmap='bwr')
            imshow_field(wfs_image,cmap='inferno')
            plt.colorbar()

            plt.subplot(2,3,4)
            plt.title('DM commands')
            plt.axis('off')
            #plt.imshow(s.reshape(self.pupil_pixels,self.pupil_pixels),cmap='bwr')
            #plt.imshow(action.reshape(self.num_act,self.num_act),cmap='bwr')
            action_map = np.zeros((self.num_act**2))
            action_map[self.actuator_mask] = action
            plt.imshow(action_map.reshape(self.num_act,self.num_act),cmap='bwr')
            plt.colorbar()

            plt.subplot(2,3,5)
            plt.title('DM shape')
            plt.axis('off')
            plt.imshow((self.dm.surface*self.aperture).shaped,cmap='bwr')
            #plt.imshow(s[:,:,1].reshape(self.num_act,self.num_act),cmap='bwr')
            plt.colorbar()

            plt.subplot(2,3,6)
            plt.title('Residual wavefront')
            plt.axis('off')
            plt.imshow(res_phase.shaped-np.mean(res_phase[self.aperture==1]),cmap='bwr')
            #plt.imshow(s.reshape(self.num_act,self.num_act),cmap='bwr')
            plt.colorbar()

            plt.draw()
            plt.pause(0.1)
        return s,r,terminate


    def reset(self, reset_turbulence=False, reset_dm=False, input_amps=None):
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
        self.science_image = np.zeros((self.focal_pixels,self.focal_pixels))
        self.science_coro_image = np.zeros((self.focal_pixels,self.focal_pixels))
        self.wavefront_variance = np.zeros((self.pupil_pixels,self.pupil_pixels))
        self.mean_res_phase = np.zeros((self.pupil_pixels,self.pupil_pixels))
        self.mean_dm_shape = np.zeros((self.pupil_pixels,self.pupil_pixels))
        self.iteration = 0
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
            self.atmosphere = MultiLayerAtmosphere(layers,scintillation=self.scintillation)

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

        s,r,terminate = self.step(np.zeros(self.num_modes))

        self.t = 0
        return(s)
	
    def compare_controllers(self, controller, num_iterations, num_trials):
        self.num_iterations = num_iterations
        for trial in range(num_trials):
            s = self.reset()
            for i in range(num_iterations):
                for cont in controllers:
                    a = cont.get_command(s)
                    s,r,t = self.env.step(a)

def make_actuator_dm(pupil_grid, num_act, D, crosstalk=0.15, cutoff=3):
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
    sigma = act_distance / (np.sqrt((-2 * np.log(crosstalk))))

    grid = make_uniform_grid(num_act, [D, D])

    #Mask out actuators outside of oversized pupil
    r_grid = np.sqrt(grid.x**2+ grid.y**2)
    actuator_mask = r_grid<=D/2
    actuator_grid = CartesianGrid(UnstructuredCoords([grid.x[actuator_mask],grid.y[actuator_mask]]))

    num_modes = np.sum(actuator_mask)
    print('\n\n\n Number of active actuators:',num_modes)
    dm_modes = make_gaussian_pokes(pupil_grid, actuator_grid, sigma, cutoff)
    dm = DeformableMirror(dm_modes)
    return(dm, actuator_mask)

