from scipy.stats import t as t_dist
from glm import glm
import numpy as np
import numpy.linalg as npl
from hypothesis import t_stat

# Order the p-values from smallest to largest
# p-value has a rank that is the index of itself in the ordered array
# critical value = (i/m)*Q where i = rank, m = number of tests, Q = FDR 
# Compare each p-value to its critical value 
# The largest p-value where p < (i/m)*Q is significant, and so are all the p-values before it 

def bh_procedure(data_4d, convolved, c = [0,1], Q):

	# Run the t_stat function from hypothesis
	beta, t, df, p = t_stat(data_4d, convolved, c = [0,1])

	# Sort the p-values
	p_sorted = np.sort(p)

	# Number of tests
	m = len(p)

	# Build critical value array
	critical_vals = []
	critical_index = None

	for x in range(len(p_sorted)):
		# i is the rank of the p-value in p_sorted
		i = p_sorted.index(p_sorted[x])
		critical_vals.append((i/m)*Q)
		if p_sorted[x] < (i/m)*Q:
			critical_index = i

	# if none of the critical values were greater than its p-value
	if critical_index == None:
		return 'No signficant tests found.'

	# Collect all the significant p-values
	significant_pvals = []
	p_index = 0

	while p_index <= critical_index:
		significant_pvals.append(p_sorted[p_index])
		p_index += 1

	# return the tests that are significant

