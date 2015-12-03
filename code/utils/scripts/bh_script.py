""" Script for the Benjamini-Hochberg function.
Run with: 
    python bh_script.py
"""

# Loading modules.
import os
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt
import nibabel as nib
import sys
import numpy.linalg as npl

# Paths. Use your own. 
pathtodata = "../../../data/ds009/sub001/"
condition_location=pathtodata+"model/model001/onsets/task001_run001/"
location_of_images="../../../images/"
sys.path.append(os.path.join(os.path.dirname('__file__'), "../functions/"))

# Load functions
from stimuli import events2neural
from event_related_fMRI_functions import hrf_single, convolution_specialized
from Image_Visualizing import present_3d, make_mask
from glm import glm
from hypothesis import t_stat
from event_related_fMRI_functions import hrf_single, convolution_specialized
from benjamini_hochberg import bh_procedure
#from bh import new_bh_procedure

# Load the image data for subject 1.
img = nib.load(pathtodata+"BOLD/task001_run001/bold.nii.gz")
data = img.get_data()
data = data[...,6:] # Knock off the first 6 observations.

cond1=np.loadtxt(condition_location+"cond001.txt")
cond2=np.loadtxt(condition_location+"cond002.txt")
cond3=np.loadtxt(condition_location+"cond003.txt")

#######################
# a. (my) convolution #
#######################

all_stimuli=np.array(sorted(list(cond2[:,0])+list(cond3[:,0])+list(cond1[:,0]))) # could also just x_s_array
my_hrf = convolution_specialized(all_stimuli,np.ones(len(all_stimuli)),hrf_single,np.linspace(0,239*2-2,239))


##################
# b. np.convolve #
##################

# Suppose that TR=2. We know this is not a good assumption.
# Also need to look into the hrf function. 
# initial needed values
TR = 2
tr_times = np.arange(0, 30, TR)
hrf_at_trs = np.array([hrf_single(x) for x in tr_times])
n_vols=data.shape[-1]

# creating the .txt file for the events2neural function
cond_all=np.row_stack((cond1,cond2,cond3))
cond_all=sorted(cond_all,key= lambda x:x[0])
np.savetxt(condition_location+"cond_all.txt",cond_all)

neural_prediction = events2neural(condition_location+"cond_all.txt",TR,n_vols)
convolved = np.convolve(neural_prediction, hrf_at_trs) # hrf_at_trs sample data
N = len(neural_prediction)  # N == n_vols == 173
M = len(hrf_at_trs)  # M == 12
np_hrf=convolved[:N]

B,t,df,p = t_stat(data, my_hrf, np.array([0,1]))



#########################
# c. Benjamini-Hochberg #
#########################
# SHIT TO DO:
# PLOT PVALUES AND THE UPPER BOUND ON A HISTOGRAM
# PLOT DIFFERENT Q VALUES AND LOOK AT THE EFFECT OF DIFFERENT FDR'S.

print("# ======= Beginning the Benjamini-Hochberg procedure now. ======= #")

"""
print("# ==== BEGIN Masking over the p-values with data ==== #")
# unravel the data
mask = nib.load(pathtodata + '/anatomy/inplane001_brain_mask.nii.gz')
mask_data = mask.get_data()
rachels_ones = np.ones((64, 64, 34))
fitted_mask = make_mask(rachels_ones, mask_data, fit = True)
print(mask_data.shape)
print(fitted_mask.shape)
mask_1d = np.ravel(fitted_mask)
print(mask_1d.shape)
smaller_p = p_vals[mask_1d == 1]

smaller_p_bh = bh_procedure(smaller_p, .2)
new_1d = np.zeros(mask_1d.shape)
new_1d[mask_1d == 1] = smaller_p_bh

new_1d_reshape = np.reshape(new_1d, fitted_mask.shape)
print(new_1d_reshape.shape)
#masked_reshape = make_mask(new_1d_reshape, fitted_mask)
plt.imshow(present_3d(new_1d_reshape),interpolation="nearest",cmap="gray")
#plt.colorbar()
plt.savefig(location_of_images+"rachels_ones.png")
plt.close()
#data_1d = np.ravel(data)

# subset into p-values array
#smaller_p = p_vals[ == 1]
"""

print("# ==== BEGIN Visualization of Masked data over original brain data ==== #")

p_vals = p.T # shape of p_vals is (139264, 1)

print("# ==== No Mask, bh_procedure ==== #")
# a fairly large false discovery rate
Q = .4
significant_pvals = bh_procedure(p_vals, Q)
#print(significant_pvals)
# Reshape significant_pvals to shape of data
reshaped_sig_p = np.reshape(significant_pvals, data.shape[:-1])
slice_reshaped_sig_p = reshaped_sig_p[...,7]
original_slice = data[...,7]

plt.imshow(slice_reshaped_sig_p)
plt.colorbar()
plt.title('Significant p-values (No mask)')
plt.savefig(location_of_images+"NOMASK_significant_p_slice.png")
plt.close()
print("# ==== END No Mask, bh_procedure ==== #")


#significant_pvals_old = bh_procedure(p_vals, fdr)

# Reshape significant_pvals to shape of data
#reshaped_sig_p_old = np.reshape(significant_pvals_old, data.shape[:-1])
#slice_reshaped_sig_p_old = reshaped_sig_p_old[...,7]
#original_slice = data[...,7]

#print("# ==== No mask, old bh_procedure")
#plt.imshow(slice_reshaped_sig_p_old)
#plt.colorbar()
#plt.title('(OLD BH FUNCTION) Significant p-values (No mask)')
#plt.savefig(location_of_images+"OLD_significant_p_slice_NOMASK.png")
#plt.close()
#print("Initial plot with NO MASK (using old bh function) done.")


print("# ==== BEGIN varying the Q value = .005 (FDR) ==== #")
Q = .005

significant_pvals = bh_procedure(p_vals, Q)

# Reshape significant_pvals
reshaped_sig_p = np.reshape(significant_pvals, data.shape[:-1])
slice_reshaped_sig_p = reshaped_sig_p[...,7]

masked_data = make_mask(original_slice, reshaped_sig_p, fit=False)

plt.imshow(present_3d(masked_data))
plt.clim(0, 1600)
plt.colorbar()
plt.title('Slice with Significant p-values (Q = .005)')
plt.savefig(location_of_images+"significant_p_slice1.png")
plt.close()
print("# ==== END plot with Q = .005 done. ==== #")

print("# ==== BEGIN varying the Q value = .05 (FDR) ==== #")
Q = .05

significant_pvals = bh_procedure(p_vals, Q)

# Reshape significant_pvals
reshaped_sig_p = np.reshape(significant_pvals, data.shape[:-1])
slice_reshaped_sig_p = reshaped_sig_p[...,7]

masked_data = make_mask(original_slice, reshaped_sig_p, fit=False)

plt.imshow(present_3d(masked_data))
plt.colorbar()
plt.title('Slice with Significant p-values (Q = .05)')
plt.savefig(location_of_images+"significant_p_slice2.png")
plt.close()
print("# ==== END plot with Q = .05 done. ==== #")


print("# ==== BEGIN varying the Q value = .10 (FDR) ==== #")
Q = .10

significant_pvals = bh_procedure(p_vals, Q)

# Reshape significant_pvals
reshaped_sig_p = np.reshape(significant_pvals, data.shape[:-1])
slice_reshaped_sig_p = reshaped_sig_p[...,7]

masked_data = make_mask(original_slice, reshaped_sig_p, fit=False)

plt.imshow(present_3d(masked_data))
plt.colorbar()
plt.title('Slice with Significant p-values (Q = .10)')
plt.savefig(location_of_images+"significant_p_slice3.png")
plt.close()
print("# ==== END plot with Q = .10 done. ==== #")

print("# ==== BEGIN the Q value = .25 (FDR) ==== #")
Q = .25
significant_pvals = bh_procedure(p_vals, Q)
slice_reshaped_sig_p = reshaped_sig_p[...,7]

masked_data = make_mask(original_slice, reshaped_sig_p, fit=False)
#masked_data2 = make_mask(original_slice, new_1d_reshape, fit=False)
plt.imshow(present_3d(masked_data))
plt.colorbar()
plt.title('Slice with Significant p-values (Q = .25)')
plt.savefig(location_of_images+"significant_p_slice4.png")
plt.close()
print("# ==== END plot with Q = .25 done. ==== #")

print("# ==== BEGIN the Q value = .5 (FDR) ==== #")
Q = .5

significant_pvals = bh_procedure(p_vals, Q)

# Reshape significant_pvals
reshaped_sig_p = np.reshape(significant_pvals, data.shape[:-1])
slice_reshaped_sig_p = reshaped_sig_p[...,7]

masked_data = make_mask(original_slice, reshaped_sig_p, fit=False)

plt.imshow(present_3d(masked_data))
plt.colorbar()
plt.title('Slice with Significant p-values (Q = .5)')
plt.savefig(location_of_images+"significant_p_slice5.png")
plt.close()
print("# ==== END plot with Q = .5 done. === #")


# =================================================================== #

print("# ==== begin Ben's stuff ==== #")
mask = nib.load(pathtodata + '/anatomy/inplane001_brain_mask.nii.gz')
mask_data = mask.get_data()
rachels_ones = np.ones((64, 64, 34))
fitted_mask = make_mask(rachels_ones, mask_data, fit = True)
fitted_mask[fitted_mask>0]=1 # corrected so there are only zeros and ones:
# plt.hist(np.ravel(fitted_mask))
# Another approach to masking
mask_new=mask_data[::2,::2,:]

assert(mask_new.shape==fitted_mask.shape)
### still going to use fitted_mask
def masking_reshape_start(data,mask):
	"""
	takes a 3 or 4d data and utilizes a mask to return a 1 or 2d reshaped output

	Input:
	------
	data: 3d *or* 4d np.array  (x,y,z) or (x,y,z,t) shape
	mask: a 3d np array  (x,y,z) shape, with values 0s and 1s (1 desired, 0 remove)

	Returns:
	--------
	reshaped: a 1d *or* 2d np.array (connected to 3d or 4d "data" input)

	"""
	assert(len(data.shape) == 3 or len(data.shape) == 4)

	if len(data.shape) == 3:
		data_1d= np.ravel(data)
		reshaped = data_1d[np.ravel(mask==1)]

	if len(data.shape) == 4:
		data_2d=data.reshape((-1,data.shape[-1]))
		reshaped = data_2d[np.ravel(mask==1),:]
	return reshaped



def masking_reshape_end(data_small,mask,off_value=0):
	"""
	takes a 1d input, utilizes a mask to convert into 3d output ()

	Notes:
	------
	mask must have same number of ones as the data_small.shape[0]

	Input:
	------
	data_small: a 1d np.array
	mask:       a 3d np array  (x,y,z) shape, with values 0s and 1s (1 desired, 0 remove), see notes
	off_value:  the value to be replaced for the non-on values of the mask


	Returns:
	--------
	data_big: 3d np.array  (x,y,z) shape

	"""
	assert(len(data_small.shape)==1)

	data_big = off_value*np.ones((mask.shape))

	data_big[mask==1]= data_small

	return data_big
p_vals_3d = p_vals.reshape((64,64,34))

to_1d = masking_reshape_start(p_vals_3d,mask_new)

for Q in [.05,.1,.25,.3,.4,.45,.49]:
	bh_1d=bh_procedure(to_1d, Q)
	to_3d = 2*masking_reshape_end(bh_1d,mask_new,.5)-1
	plt.imshow(present_3d(to_3d),interpolation='nearest', cmap='seismic')
	plt.title(Q)
	plt.colorbar()
	plt.figure()

	#plt.figure()
	test = to_3d
	# maybe some clustering or smoothing?


	off = np.max(test)
	for i in 1 + np.arange(62):
		for j in 1 + np.arange(62):
			for k in 1 + np.arange(32):
				if np.sum(to_3d[(i - 1):(i + 2),(j - 1):(j + 2),(k - 1):(k + 2)] < 0) < 10 and to_3d[i,j,k] < 0:
					test[i,j,k] = off


	plt.imshow(present_3d(test),interpolation='nearest', cmap='seismic')
	plt.title(str(Q)+ " mini-smoothing")
	plt.savefig(location_of_images+"ben_plot_bh_"+str(Q)+".png")
	plt.colorbar()