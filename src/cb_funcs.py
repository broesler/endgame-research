#!/Users/bernardroesler/anaconda3/bin/python3
#==============================================================================
#     File: cb_funcs.py
#  Created: 06/14/2018, 17:29
#   Author: Bernie Roesler
#
"""
  Description: Helper functions
"""
#==============================================================================

import numpy as np

def Nstats(X, y):
    """Return tuple of fractions for given dataset."""
    N = X.shape[0] # total number of examples
    Fs = y[(y.success == 1) & (y.failure == 0)].shape[0] / N # fraction success
    Ff = y[(y.success == 0) & (y.failure == 1)].shape[0] / N # fraction failure
    Fu = y[(y.success == 0) & (y.failure == 0)].shape[0] / N # fraction unknown
    return np.array([N, Fs, Ff, Fu])

#==============================================================================
#==============================================================================
