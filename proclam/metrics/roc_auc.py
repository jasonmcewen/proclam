"""
The AUC of the ROC
"""

from __future__ import absolute_import
__all__ = ['ROCAUC']

import numpy as np

class ROCAUC(object):

    def __init__(self, scheme=None):
        """
        An object that evaluates the AUC of the ROC

        Parameters
        ----------
        scheme: string
            the name of the metric
        """
        super(ROCAUC, self).__init__(scheme)
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
        #   loop over thresholds: [maybe want this to be done by a helper in proclam.metrics.util]
        #     call proclam.metrics.util.binarize
        #     call proclam.metrics.util.det_to_cm
        #     call proclam.metrics.util.cm_to_rate
        #     save per-class per-threshold TPR and FPR in an array
        #   call proclam.metrics.util.auc(FPR, TPR)
        # average the aucs (ideally call proclam.metrics.util.average when it exists) using weight (equal per-class if none)
        # return the auc
        pass
