import numpy as np
import copy
from hcipy.optics.deformable_mirror import make_gaussian_pokes
from hcipy.field import make_uniform_grid
from hcipy.field.cartesian_grid import CartesianGrid
from hcipy.field.coordinates import UnstructuredCoords

def make_actuator_positions(num_actuators_across_pupil, actuator_spacing, x_tilt=0, y_tilt=0, z_tilt=0, oversizing=1):
	'''Make actuator positions using the BMC convention.
	Parameters
	----------
	num_actuators_across_pupil : integer
		The number of actuators across the pupil. The total number of actuators will be this number squared.
	actuator_spacing : scalar
		The spacing between actuators before tilting the deformable mirror.
	x_tilt : scalar
		The tilt of the deformable mirror around the x-axis in radians.
	y_tilt : scalar
		The tilt of the deformable mirror around the y-axis in radians.
	z_tilt : scalar
		The tilt of the deformable mirror around the z-axis in radians.
	Returns
	-------
	Grid
		The actuator positions.
	'''
	extent = actuator_spacing * num_actuators_across_pupil
	grid = make_uniform_grid(num_actuators_across_pupil, [extent, extent]).scaled(np.cos([y_tilt, x_tilt]))

	mask = np.sqrt(grid.x**2+grid.y**2)<(oversizing*extent)
	grid = CartesianGrid(UnstructuredCoords([grid.x[mask], grid.y[mask]]))
	if z_tilt == 0:
		return grid

	grid = grid.rotated(z_tilt)
	return grid, mask

def make_cutoff_gaussian_influence_functions(pupil_grid, num_actuators_across_pupil, actuator_spacing, crosstalk=0.15, cutoff=3, x_tilt=0, y_tilt=0, z_tilt=0, oversizing=1):
	'''Create influence functions with a Gaussian profile.
	The default value for the crosstalk is representative for Boston Micromachines DMs.
	Parameters
	----------
	pupil_grid : Grid
		The grid on which to calculate the influence functions.
	num_actuators_across_pupil : integer
		The number of actuators across the pupil. The total number of actuators will be this number squared.
	actuator_spacing : scalar
		The spacing between actuators before tilting the deformable mirror.
	crosstalk : scalar
		The amount of crosstalk between the actuators. This is defined as the value of the influence function
		at a nearest-neighbour actuator.
	cutoff : scalar
		The distance from the center of the actuator, as a fraction of the actuator spacing, where the
		influence function is truncated to zero.
	x_tilt : scalar
		The tilt of the deformable mirror around the x-axis in radians.
	y_tilt : scalar
		The tilt of the deformable mirror around the y-axis in radians.
	z_tilt : scalar
		The tilt of the deformable mirror around the z-axis in radians.
	Returns
	-------
	ModeBasis
		The influence functions for each of the actuators.
	'''
	actuator_positions, mask = make_actuator_positions(num_actuators_across_pupil, actuator_spacing, oversizing=oversizing)

	# Stretch and rotate pupil_grid to correct for tilted DM
	evaluated_grid = pupil_grid.scaled(1 / np.cos([y_tilt, x_tilt])).rotated(-z_tilt)

	sigma = actuator_spacing / (np.sqrt((-2 * np.log(crosstalk))))
	cutoff = actuator_spacing / sigma * cutoff

	pokes = make_gaussian_pokes(evaluated_grid, actuator_positions, sigma, cutoff)
	pokes.transformation_matrix /= np.cos(x_tilt) * np.cos(y_tilt)
	pokes.grid = pupil_grid

	return pokes, mask
    