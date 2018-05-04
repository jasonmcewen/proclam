"""
The AUC of the ROC
"""

from __future__ import absolute_import
__all__ = ['ROCAUC']

import numpy as np
from proclam.metrics import util

class ROCAUC(object):

	def __init__(self, scheme=None):
		"""
		An object that evaluates the AUC of the ROC

		Parameters
		----------
		scheme: string
			the name of the metric
		"""
		self.scheme = scheme

	def evaluate(self, prediction, truth, thresholds=100):
		"""
		Evaluates the AUC of the ROC

		Parameters
		----------
		prediction: numpy.ndarray, float
			predicted class probabilities
		truth: numpy.ndarray, int
			true classes
		thresholds: numpy.ndarray, float or int
			probability thresholds or number of thresholds
		[maybe a keyword for weighting scheme?]

		Returns
		-------
		roc_auc: float
			value of the auc of the roc
		"""
		# check if threshold is array of floats 0<=t<=1
		# if not, use np.linspace to make such an array with `thresholds` # of points

		# loop over classes: [maybe want this to be done by a helper in proclam.metrics.util]
		#	loop over thresholds: [maybe want this to be done by a helper in proclam.metrics.util]
		#	  call proclam.metrics.util.binarize
		#	  call proclam.metrics.util.det_to_cm
		#	  call proclam.metrics.util.cm_to_rate
		#	  save per-class per-threshold TPR and FPR in an array
		#	call proclam.metrics.util.auc(FPR, TPR)
		# average the aucs (ideally call proclam.metrics.util.average when it exists) using weight (equal per-class if none)
		# return the auc

		if not hasattr(thresholds, '__len__'):
			thresholds = np.linspace(0,1,thresholds)
			
		ratesmat = []; auc = []
		nclass = np.unique(truth)
		for c in nclass:
			fpr,tpr = np.array([]),np.array([])
			for t in thresholds:
				binaryprobs = util.binarize(prediction, t, c)
				
				# convert probs and truth into format w/ only yes/no options
				binaryprobs_onecol = np.zeros(np.shape(binaryprobs)[0])
				binaryprobs_onecol[:] = binaryprobs[:,c]
				
				binarytruth = np.zeros(len(truth),dtype='int')
				binarytruth[truth == c] = 1
				binarytruth[truth != c] = 0
				
				confusemat = util.det_to_cm(binaryprobs_onecol.astype(int),binarytruth,
											per_class_norm=True,vb=False)
				ratesmat = util.cm_to_rate(confusemat,vb=False)
				fpr = np.append(fpr,confusemat[0,1]); tpr = np.append(tpr,confusemat[0,0])
				#if np.abs(t - 0.51) < 0.001: import pdb; pdb.set_trace()
			auc += [util.AUC(tpr,fpr)]

			
		return auc
