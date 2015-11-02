import numpy as np
import itertools

def smooth_easy(data_4d):
	"""
	Return a 'smoothed' version of data_4d.

	Parameters
	----------
	data_4d : numpy array of 4 dimensions 
        The image data of one subject

	Returns
    -------
    smooth_results : array of the smoothed data from data_4d (same dimensions but super-voxels will be
    					indicated by the same number)
	"""
	return None
	# for i in range 1 to (total number of voxels - 1)
	# for i in range(1, ):
		# take the average with two neighbors from each side, replace value at those indices with new avg.
def smooth_advanced(data_4d, factor=4):
	"""
	Return a 'smoothed' version of data_4d.

	Parameters
	----------
	data_4d : numpy array of 4 dimensions 
        The image data of one subject
    factor : the factor by which to reduce data_4d into factor-cubed chunks
    	Smoothing factor

    Note that the number of voxels may not be evenly divisble into factor-cubed chunks.

    Returns
    -------
    smooth_results : array of the smoothed data from data_4d
	"""
	# Cut down data to be evenly split into 4 x 4 x 4 chunks (use mod %)
	x_size = data_4d.shape[0]//factor
	y_size = data_4d.shape[1]//factor
	z_size = data_4d.shape[2]//factor

	# Initialize a return list of zeros, same size as data_4d
	smooth_results = np.zeros(data_4d.shape)

	# Iterate data_4d by 4 x 4 x 4 (defaut factor size) chunks
	for x, y, z in itertools.product(*map(range, (x_size, y_size, z_size))):
		# Extract the chunk of data to average over
		data_chunk = data_4d[x*factor : x*factor + factor, y*factor : y*factor + factor, z*factor : z*factor + 1]
		# Flatten data_chunk, then find the mean of this array
		data_mean = np.mean(data_chunk)
		# Insert the value data_mean for the indices of the chunk given by data_chunk into smooth_results
		smooth_results[x*factor : x*factor + factor, y*factor : y*factor + factor, z*factor : z*factor + 1] = data_mean

	# Need a way to average the values that don't get touched upon??? Maybe??

	return smooth_results	

	