""" Script for smooth function.
Run with: 
    python smooth_script.py

in the scripts directory
"""

import numpy as np
import itertools
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import nibabel as nib
import os
import sys
import math

# Relative path to subject 1 data
pathtodata = "../../../data/ds009/sub001/"
condition_location=pathtodata+"model/model001/onsets/task001_run001/"
location_of_images="../../../images/"

#sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))
sys.path.append("../functions")

# Load events2neural from the stimuli module.
#from stimuli import events2neural
#from event_related_fMRI_functions import hrf_single, convolution_specialized

# Load our GLM functions. 
#from glm import glm, glm_diagnostics, glm_multiple

# Load smoothing function
from smooth import smoothvoxels
from Image_Visualizing import present_3d

# Load the image data for subject 1.
img = nib.load(pathtodata+"BOLD/task001_run001/bold.nii.gz")
data = img.get_data()
data = data[...,6:] # Knock off the first 6 observations.

############################
# a. (my) smoothing plots  #
############################
print("# ===== Begin smoothing plots ===== #")

# Kind of arbitrary chosen time
time = 7
original_slice = data[..., 7]

print("Making plot of original slice")
plt.imshow(present_3d(original_slice))
plt.colorbar()
plt.title('Original Slice')
plt.clim(0,1600)
plt.savefig(location_of_images+"original_slice.png")
plt.close()

# ====== Play around with values of sigma ====== #
print("Making plot with sigma = .25")
sigma0 = .25
smoothed_slice0 = smoothvoxels(data, sigma0, time)
plt.imshow(present_3d(smoothed_slice0))
plt.colorbar()
plt.title('Smoothed Slice where sigma = .25')
plt.clim(0,1600)
plt.savefig(location_of_images+"smoothed_slice0.png")
plt.close()

print("Making plot with sigma = .5")
sigma1 = .5
smoothed_slice1 = smoothvoxels(data, sigma1, time)
plt.imshow(present_3d(smoothed_slice1))
plt.colorbar()
plt.title('Smoothed Slice where sigma = .5')
plt.clim(0,1600)
plt.savefig(location_of_images+"smoothed_slice1.png")
plt.close()

print("Making plot with sigma = 1.5")
sigma2 = 1.5
smoothed_slice2 = smoothvoxels(data, sigma2, time)
plt.imshow(present_3d(smoothed_slice2))
plt.colorbar()
plt.title('Smoothed Slice where sigma = 1.5')
plt.clim(0,1600)
plt.savefig(location_of_images+"smoothed_slice2.png")
plt.close()

print("Making plot with sigma = 2.5")
sigma3 = 2.5
smoothed_slice3 = smoothvoxels(data, sigma3, time)
plt.imshow(present_3d(smoothed_slice3))
plt.colorbar()
plt.title('Smoothed Slice where sigma = 2.5')
plt.clim(0,1600)
plt.savefig(location_of_images+"smoothed_slice3.png")
plt.close()

print("# ===== Plots completed ===== #")