import numpy as np
import itertools

from scipy.ndimage.filters import gaussian_filter



def smooth(data_4d, sigma, time):

	"""
	Return a 'smoothed' version of data_4d.

	Parameters
	----------
	data_4d : numpy array of 4 dimensions 
        The image data of one subject


    sigma : width of normal gaussian curve

    time : time slice (4th dimension)

	Returns
    -------
    smooth_results : array of the smoothed data from data_4d (same dimensions but super-voxels will be
    					indicated by the same number) in time slice indicated.
	"""
	time_slice = data_4d[..., time]
	smooth_results = scipy.ndimage.filters.gaussian_filter(time_slice, sigma)
	return smooth_results

