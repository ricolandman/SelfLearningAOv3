from hcipy.wavefront_sensing.pyramid import *
class CustomPyramidWavefrontSensorOptics(WavefrontSensorOptics):
	'''The optical elements for a pyramid wavefront sensor.
	Parameters
	----------
	input_grid : Grid
		The grid on which the input wavefront is defined.
	separation : scalar
		The separation between the pupils. The default takes the input grid extent as separation.
	wavelength_0 : scalar
		The reference wavelength that determines the physical scales.
	q : scalar
		The focal plane oversampling coefficient. The default uses the minimal required sampling.
	refractive_index : callable
		A callable that returns the refractive index as function of wavelength.
		The default is a refractive index of 1.5.
	num_airy : scalar
		The radius of the focal plane spatial filter in units of lambda/D at the reference wavelength.
	'''
	def __init__(self, input_grid, num_output_pixels, separation=None, wavelength_0=1, q=None, num_airy=None, refractive_index=lambda x: 1.5):
		if not input_grid.is_regular:
			raise ValueError('The input grid must be a regular grid.')

		self.input_grid = input_grid
		D = np.max(input_grid.delta * (input_grid.shape - 1))

		if separation is None:
			separation = D

		# Oversampling necessary to see all frequencies in the output wavefront sensor plane
		qmin = max(2 * separation / D, 1)
		if q is None:
			q = qmin
		elif q < qmin:
			raise ValueError('The requested focal plane sampling is too low to sufficiently sample the wavefront sensor output.')

		if num_airy is None:
			self.num_airy = np.max(input_grid.shape - 1) / 2
		else:
			self.num_airy = num_airy

		self.focal_grid = make_focal_grid(q, self.num_airy, reference_wavelength=wavelength_0, pupil_diameter=D, focal_length=1)
		self.output_grid = make_pupil_grid(num_output_pixels, qmin * D)

		# Make all the optical elements
		self.spatial_filter = Apodizer(circular_aperture(2 * self.num_airy * wavelength_0 / D)(self.focal_grid))
		pyramid_surface = -separation / (2 * (refractive_index(wavelength_0) - 1)) * (np.abs(self.focal_grid.x) + np.abs(self.focal_grid.y))
		self.pyramid = SurfaceApodizer(Field(pyramid_surface, self.focal_grid), refractive_index)

		# Make the propagators
		self.pupil_to_focal = FraunhoferPropagator(input_grid, self.focal_grid)
		self.focal_to_pupil = FraunhoferPropagator(self.focal_grid, self.output_grid)

	def forward(self, wavefront):
		'''Propagates a wavefront through the pyramid wavefront sensor.
		Parameters
		----------
		wavefront : Wavefront
			The input wavefront that will propagate through the system.
		Returns
		-------
		wf_wfs : Wavefront
			The output wavefront.
		'''
		wf_focus = self.pupil_to_focal.forward(wavefront)
		wf_pyramid = self.pyramid.forward(self.spatial_filter.forward(wf_focus))
		wf_wfs = self.focal_to_pupil.forward(wf_pyramid)

		return wf_wfs

	def backward(self, wavefront):
		'''Propagates a wavefront backwards through the pyramid wavefront sensor.
		Parameters
		----------
		wavefront : Wavefront
			The input wavefront that will propagate through the system.
		Returns
		-------
		wf_pupil : Wavefront
			The output wavefront.
		'''
		wf_focus = self.focal_to_pupil.backward(wavefront)
		wf_pyramid = self.pyramid.backward(self.spatial_filter.backward(wf_focus))
		wf_pupil = self.pupil_to_focal.backward(wf_pyramid)

		return wf_pupil