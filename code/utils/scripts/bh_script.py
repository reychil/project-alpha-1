""" Script for benjamini-hochberg function.
Run with: 
    python bh_script.py

in the scripts directory
"""

from scipy.stats import t as t_dist
from glm import glm
import numpy as np
import numpy.linalg as npl
from hypothesis import t_stat

# Relative path to subject 1 data
pathtodata = "../../../data/ds009/sub001/"
condition_location=pathtodata+"model/model001/onsets/task001_run001/"
location_of_images="../../../images/"

#sys.path.append(os.path.join(os.path.dirname(__file__), "../functions/"))
sys.path.append("../functions")

# Load benjamini-hochberg function
from benjamini_hochberg import bh_procedure

# Load the image data for subject 1.
img = nib.load(pathtodata+"BOLD/task001_run001/bold.nii.gz")
data = img.get_data()
data = data[...,6:] # Knock off the first 6 observations.

