""" Tests for bh_procedure in benjamini_hochberg module

Run at the project directory with:
    nosetests code/utils/tests/test_bh.py
"""

# Loading modules.
import numpy as np
import itertools
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import nibabel as nib
import os
import sys
from numpy.testing import assert_almost_equal
from nose.tools import assert_not_equals
from hypothesis import t_stat

# Path to the subject 009 fMRI data used in class. 
pathtoclassdata = "data/ds114/"

# Add path to functions to the system path.
sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))

# Load our benjamini-hochberg function
from benjamini_hochberg import bh_procedure

def test_bh():
	img = nib.load(pathtoclassdata + "ds114_sub009_t2r1.nii")
    data = img.get_data()[..., 4:]
    # Read in the convolutions. 
    convolved = np.loadtxt(pathtoclassdata + "ds114_sub009_t2r1_conv.txt")[4:]
    # Create design matrix. 
    
    beta,t,df,p = t_stat(data, convolved,[1,1])
    beta2, t2,df2,p2 = t_stat(data, convolved,[0,1])
