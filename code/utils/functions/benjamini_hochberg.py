from scipy.stats import t as t_dist
from glm import glm
import numpy as np
import numpy.linalg as npl
#from hypothesis import t_stat

# Order the p-values from smallest to largest
# p-value has a rank that is the index of itself in the ordered array
# critical value = (i/m)*Q where i = rank, m = number of tests, Q = FDR 
# Compare each p-value to its critical value 
# The largest p-value where p < (i/m)*Q is significant, and so are all the p-values before it 

def bh_procedure(p_vals, Q):
	"""
	Return an array (mask) of the significant, valid tests
		out of the p-values. not significant p-values are denoted by ones.

	Parameters
	----------
    p_vals: p-values from the t_stat function (1-dimensional array)

    Q: The false discovery rate 


	Returns
    -------
    significant_pvals : 1-d array of p-values of tests that are
    	deemed significant, denoted by 1's and p-values

    Note: You will have to reshape the output to the shape of the data set.
	"""
	# k is Q/m where m = len(p_vals)
	k = Q/len(p_vals)

	# Multiply an array of rank values by k
	upper = k*np.array(0, 1 + len(p_vals))





	# Sort the p-values
	p_sorted = np.sort(p_vals)

	# ordering of the original p_vals
	p_indices = np.argsort(p_vals)

	# Number of tests
	m = len(p_vals)

	# Build critical value array
	#critical_vals = []
	critical_index = None

	for x in range(len(p_sorted)):

		# rank of the p-value in p_sorted
		rank = p_sorted.index(p_sorted[x])

		if p_sorted[x] < (rank/m)*Q:
			critical_index = i



	# if none of the critical values were greater than its p-value
	if critical_index == None:
		return 'No signficant tests found.'

	index = 0
	significant_pvals = np.ones(m)

	for p in p_indices:
		significant_pvals[p] = p_sorted[index]
		if index >= critical_index:
			break

	# return the tests that are significant
	return significant_pvals
