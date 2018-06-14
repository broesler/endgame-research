#!/Users/bernardroesler/anaconda3/envs/insight/bin/python3
#==============================================================================
#     File: train_model.py
#  Created: 06/13/2018, 15:11
#   Author: Bernie Roesler
#
"""
  Description: Create, train, and cross-validate initial model
"""
#==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc as mplrc
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

plt.ion()

# IMPORT THE DATA!!
from build_features import df, Ntot

feat_cols = ['age_at_exit', 'milestones', 'latitude', 'longitude', 'offices',
             'products', 'funding_rounds', 'investment_rounds',
             'invested_companies', 'acq_before_exit', 'investors',
             'investors_per_round', 'funding_per_round', 'experience']
labels = ['label']

#------------------------------------------------------------------------------ 
#        Data Statistics
#------------------------------------------------------------------------------
N = df.shape[0]
Nf = df[df.label == 0].shape[0] / N      # fraction failure
Ne = df[df.label == 1].shape[0] / N      # fraction success
Nu = df[df.label == 2].shape[0] / N      # fraction unknown

#------------------------------------------------------------------------------ 
#        Test/Train Split
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------ 
#        Train the Model
#------------------------------------------------------------------------------
rfc = RandomForestClassifier(n_estimators=10)
rfc = rfc.fit(df[feat_cols], df[labels])

#==============================================================================
#==============================================================================
