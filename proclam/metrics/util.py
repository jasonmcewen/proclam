"""
metrics utilities
"""

import numpy as np
from sklearn.metrics import roc_auc_score

def AUC(pvals, truth, pthresh=0.5):
	"""
	Measures the AUC for a probability
	threshold given by pthresh
	"""

	assert type(pvals) == np.ndarray
	assert type(truth) == np.ndarray

	if len(np.unique(truth[pvals > pthresh])) == 1:
		print('Warning : only one class with P > %.2f cut'%pthresh)
		return np.nan
	
	return roc_auc_score(truth[pvals > pthresh],
						 pvals[pvals > pthresh])
