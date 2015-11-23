import numpy as np
import itertools
import scipy.ndimage
import math
from scipy.ndimage.filters import gaussian_filter



def smoothvoxels(data_4d, sigma, time):

	"""
	Return a 'smoothed' version of data_4d.

	Parameters
	----------
	data_4d : numpy array of 4 dimensions 
        The image data of one subject


    fwhm : width of normal gaussian curve

    time : time slice (4th dimension)

	Returns
    -------
    smooth_results : array of the smoothed data from data_4d (same dimensions but super-voxels will be
    					indicated by the same number) in time slice indicated.
	"""
	time_slice = data_4d[..., time]
	fwhm = 2*math.sqrt(2*math.log(2)*sigma)
	smooth_results = scipy.ndimage.filters.gaussian_filter(time_slice, fwhm)
	return smooth_results

